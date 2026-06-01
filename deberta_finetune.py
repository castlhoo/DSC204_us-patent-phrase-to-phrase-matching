"""
US Patent Phrase Similarity — DeBERTa-v3-base Fine-tuning
회귀 모델: anchor vs target 유사도 점수(0~1) 예측
Metric: Pearson correlation

실행:
  python deberta_finetune.py
나중에 fold 늘리려면 CFG["train_folds"] = [0,1,2,3,4] 로 변경
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["HF_HOME"] = os.path.expanduser("~/.hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.hf_cache/hub")

import gc
import zipfile, io, random, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    DebertaV2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
import scipy.stats as stats

# ── 설정 ──────────────────────────────────────────────────────
CFG = {
    "model_name"  : "microsoft/deberta-v3-large",
    "max_length"  : 96,
    "batch_size"  : 8,
    "grad_accum"  : 4,
    "epochs"      : 4,
    "lr"          : 2e-5,
    "warmup_ratio": 0.1,
    "seed"        : 42,
    "n_folds"     : 5,
    "train_folds" : [0,1,2,3,4],
    "data_path"   : "data/us-patent-phrase-to-phrase-matching.zip",
    "device"      : "cuda" if torch.cuda.is_available() else "cpu",
    "ckpt_dir"    : "checkpoints",
}

os.makedirs(CFG["ckpt_dir"], exist_ok=True)
PROGRESS_FILE = os.path.join(CFG["ckpt_dir"], "progress.json")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(CFG["seed"])
torch.backends.cudnn.benchmark = True
# Colab+PyTorch2.6에서 AMP가 수렴을 깨뜨려 pure float32 사용.
# TF32도 끔(DeBERTa attention 정밀도 보호).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print(f"Device: {CFG['device']}", flush=True)

# ── 데이터 로드 ───────────────────────────────────────────────
z     = zipfile.ZipFile(CFG["data_path"])
train = pd.read_csv(io.BytesIO(z.read("train.csv")))
test  = pd.read_csv(io.BytesIO(z.read("test.csv")))
sub   = pd.read_csv(io.BytesIO(z.read("sample_submission.csv")))

print(f"Train: {train.shape}, Test: {test.shape}", flush=True)

# ── CPC 섹션 설명 매핑 ──────────────────────────────────────
CPC_SECTION = {
    'A': 'human necessities',
    'B': 'performing operations transporting',
    'C': 'chemistry metallurgy',
    'D': 'textiles paper',
    'E': 'fixed constructions',
    'F': 'mechanical engineering lighting heating',
    'G': 'physics',
    'H': 'electricity',
    'Y': 'general tagging new developments',
}

def expand_cpc(code):
    section = str(code)[0].upper() if pd.notna(code) and len(str(code)) > 0 else ''
    return CPC_SECTION.get(section, str(code))

# ── 토크나이저 ────────────────────────────────────────────────
tokenizer = DebertaV2Tokenizer.from_pretrained(CFG["model_name"])

# ── Dataset (pre-tokenize) ────────────────────────────────────
class PatentDataset(Dataset):
    def __init__(self, df, is_test=False, augment=False):
        self.is_test = is_test
        if augment and not is_test:
            swapped = df.copy()
            swapped["anchor"] = df["target"].values
            swapped["target"] = df["anchor"].values
            df = pd.concat([df, swapped], ignore_index=True)
        else:
            df = df.reset_index(drop=True)

        texts_a = (df["context"].map(expand_cpc) + " " + df["anchor"]).tolist()
        texts_b = df["target"].tolist()
        print(f"  토큰화 중... ({len(texts_a)}개)", flush=True)
        enc = tokenizer(
            texts_a, texts_b,
            max_length=CFG["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids")
        if not is_test:
            self.labels = torch.tensor(df["score"].values, dtype=torch.float)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx]}
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        if self.is_test:
            return item
        return item, self.labels[idx]

# ── 모델 ──────────────────────────────────────────────────────
class PatentModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone   = AutoModel.from_pretrained(model_name)
        # gradient_checkpointing 제거: PyTorch 2.6 autocast 충돌 회피. A100이면 불필요.
        hidden          = self.backbone.config.hidden_size
        self.dropout    = nn.Dropout(0.1)
        self.regressor  = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.backbone(**kwargs)
        cls = out.last_hidden_state[:, 0, :]
        return torch.sigmoid(self.regressor(self.dropout(cls))).squeeze(-1)

# ── 학습 함수 ─────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    criterion  = nn.MSELoss()

    for step, (batch, labels) in enumerate(loader):
        input_ids      = batch["input_ids"].to(CFG["device"])
        attention_mask = batch["attention_mask"].to(CFG["device"])
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(CFG["device"])
        labels = labels.to(CFG["device"])

        with torch.cuda.amp.autocast():
            preds = model(input_ids, attention_mask, token_type_ids)
            loss  = criterion(preds, labels) / CFG["grad_accum"]

        scaler.scale(loss).backward()

        if (step + 1) % CFG["grad_accum"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * CFG["grad_accum"]
        if step % 100 == 0:
            bb_grad = 0.0
            for p in model.backbone.parameters():
                if p.grad is not None:
                    bb_grad = p.grad.norm().item()
                    break
            print(f"  step {step}/{len(loader)}  loss={total_loss/(step+1):.4f}  bb_grad={bb_grad:.6f}", flush=True)

    return total_loss / len(loader)

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        input_ids      = batch["input_ids"].to(CFG["device"])
        attention_mask = batch["attention_mask"].to(CFG["device"])
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(CFG["device"])
        out = model(input_ids, attention_mask, token_type_ids)
        preds.append(out.cpu().numpy())
    return np.concatenate(preds)

# ── 체크포인트 ────────────────────────────────────────────────
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_folds": {}}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

# ── K-Fold 학습 ───────────────────────────────────────────────
# 점수 0,0.25,0.5,0.75,1.0 → stratify용 정수 레이블
train["label"] = (train["score"] * 4).round().astype(int)

skf         = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
fold_splits = list(skf.split(train, train["label"]))

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))

test_ds     = PatentDataset(test, is_test=True)
test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"] * 2, shuffle=False, num_workers=0, pin_memory=False)

progress        = load_progress()
completed_folds = progress.get("completed_folds", {})
n_folds_used    = 0

# 완료된 fold 로드
for fold_str, fold_info in completed_folds.items():
    fold = int(fold_str)
    if fold not in CFG["train_folds"]:
        continue
    oof_path  = f"{CFG['ckpt_dir']}/fold{fold}_oof.npy"
    test_path = f"{CFG['ckpt_dir']}/fold{fold}_test.npy"
    if os.path.exists(oof_path) and os.path.exists(test_path):
        _, val_idx = fold_splits[fold]
        oof_preds[val_idx] = np.load(oof_path)
        test_preds        += np.load(test_path)
        n_folds_used      += 1
        print(f"Fold {fold} 체크포인트 로드 (pearson={fold_info['pearson']:.4f})", flush=True)

# 남은 fold 학습
for fold in CFG["train_folds"]:
    if str(fold) in completed_folds:
        print(f"Fold {fold} 이미 완료, 건너뜀", flush=True)
        continue

    tr_idx, val_idx = fold_splits[fold]
    print(f"\n{'='*40}")
    print(f" FOLD {fold}  (train={len(tr_idx)}, val={len(val_idx)})")
    print(f"{'='*40}", flush=True)

    tr_df  = train.iloc[tr_idx]
    val_df = train.iloc[val_idx]

    tr_ds      = PatentDataset(tr_df, augment=True)
    val_ds     = PatentDataset(val_df)
    tr_loader  = DataLoader(tr_ds,  batch_size=CFG["batch_size"],     shuffle=True,  num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False, num_workers=0, pin_memory=False)

    model        = PatentModel(CFG["model_name"]).to(CFG["device"])
    optimizer    = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)
    total_steps  = len(tr_loader) // CFG["grad_accum"] * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler()

    # 에포크 체크포인트 복구
    start_epoch     = 0
    best_pearson    = -1.0
    epoch_ckpt_path = f"{CFG['ckpt_dir']}/fold{fold}_latest.pt"
    best_ckpt_path  = f"{CFG['ckpt_dir']}/fold{fold}_best.pt"

    if os.path.exists(epoch_ckpt_path):
        ckpt = torch.load(epoch_ckpt_path, map_location=CFG["device"])
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch  = ckpt["epoch"] + 1
        best_pearson = ckpt["best_pearson"]
        print(f"  에포크 {start_epoch+1} 부터 재시작 (best_pearson={best_pearson:.4f})", flush=True)

    for epoch in range(start_epoch, CFG["epochs"]):
        print(f"\n[Fold {fold} | Epoch {epoch+1}/{CFG['epochs']}]", flush=True)
        train_loss = train_one_epoch(model, tr_loader, optimizer, scheduler, scaler)

        val_preds  = predict(model, val_loader)
        val_labels = val_df["score"].values
        pearson    = stats.pearsonr(val_labels, val_preds)[0]
        print(f"  train_loss={train_loss:.4f}  val_pearson={pearson:.4f}", flush=True)

        if pearson > best_pearson:
            best_pearson = pearson
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  → Best 모델 저장 (pearson={best_pearson:.4f})", flush=True)

        torch.save({
            "epoch"       : epoch,
            "model"       : model.state_dict(),
            "optimizer"   : optimizer.state_dict(),
            "scheduler"   : scheduler.state_dict(),
            "scaler"      : scaler.state_dict(),
            "best_pearson": best_pearson,
        }, epoch_ckpt_path)

    # Best 모델로 최종 예측
    model.load_state_dict(torch.load(best_ckpt_path, map_location=CFG["device"]))
    fold_oof  = predict(model, val_loader)
    fold_test = predict(model, test_loader)

    np.save(f"{CFG['ckpt_dir']}/fold{fold}_oof.npy",  fold_oof)
    np.save(f"{CFG['ckpt_dir']}/fold{fold}_test.npy", fold_test)

    oof_preds[val_idx] = fold_oof
    test_preds        += fold_test
    n_folds_used      += 1

    completed_folds[str(fold)] = {"pearson": float(best_pearson)}
    progress["completed_folds"] = completed_folds
    save_progress(progress)

    if os.path.exists(epoch_ckpt_path):
        os.remove(epoch_ckpt_path)
    if os.path.exists(best_ckpt_path):
        os.remove(best_ckpt_path)

    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nFold {fold} 완료. Best pearson={best_pearson:.4f}", flush=True)

# ── OOF 점수 ─────────────────────────────────────────────────
mask     = np.array([str(f) in completed_folds for f in range(CFG["n_folds"])
                     for _ in [fold_splits[f][1]]], dtype=bool)
used_idx = np.concatenate([fold_splits[f][1] for f in CFG["train_folds"]
                           if str(f) in completed_folds])
oof_pearson = stats.pearsonr(train.iloc[used_idx]["score"].values, oof_preds[used_idx])[0]
print(f"\nOOF Pearson: {oof_pearson:.4f}")

# ── 제출 파일 ─────────────────────────────────────────────────
sub["score"] = test_preds / n_folds_used
sub["score"] = sub["score"].clip(0, 1)
sub.to_csv("predictions.csv", index=False)
print("predictions.csv 저장 완료!")
print(sub.head())
