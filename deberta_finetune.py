"""
US Patent Phrase Similarity — DeBERTa-v3-large Fine-tuning v2
개선: Target Groupby + EMA + BI-LSTM + AWP(ep2~) + Differential LR + max_len 192
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["HF_HOME"] = os.path.expanduser("~/.hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.hf_cache/hub")

import gc
import zipfile, io, random, json
from collections import defaultdict
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
    "model_name"          : "microsoft/deberta-v3-large",
    "max_length"          : 192,
    "batch_size"          : 4,
    "grad_accum"          : 8,
    "epochs"              : 3,
    "lr"                  : 2e-5,
    "head_lr"             : 1e-3,
    "warmup_ratio"        : 0.1,
    "seed"                : 42,
    "n_folds"             : 5,
    "train_folds"         : [0,1,2,3,4],
    "data_path"           : "data/us-patent-phrase-to-phrase-matching.zip",
    "device"              : "cuda" if torch.cuda.is_available() else "cpu",
    "ckpt_dir"            : "checkpoints",
    "ema_decay"           : 0.999,
    "awp_lr"              : 1e-4,
    "awp_eps"             : 1e-2,
    "awp_start_epoch"     : 1,        # epoch 2부터 AWP (0-indexed)
    "max_grouped_targets" : 5,
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

# ── Target Groupby ────────────────────────────────────────────
def build_group_map(df):
    gmap = defaultdict(list)
    for _, row in df.iterrows():
        gmap[(row['anchor'], row['context'])].append(row['target'])
    for key in gmap:
        random.shuffle(gmap[key])
    return dict(gmap)

def get_grouped_target(target, anchor, context, gmap):
    others = [t for t in gmap.get((anchor, context), []) if t != target]
    others = others[:CFG['max_grouped_targets']]
    return (target + '; ' + '; '.join(others)) if others else target

# ── Dataset (pre-tokenize + groupby) ─────────────────────────
class PatentDataset(Dataset):
    def __init__(self, df, is_test=False, augment=False, gmap=None):
        self.is_test = is_test
        if augment and not is_test:
            swapped = df.copy()
            swapped["anchor"] = df["target"].values
            swapped["target"] = df["anchor"].values
            df = pd.concat([df, swapped], ignore_index=True)
        else:
            df = df.reset_index(drop=True)

        texts_a = (df["context"].map(expand_cpc) + " " + df["anchor"]).tolist()
        if gmap is not None:
            texts_b = [get_grouped_target(r['target'], r['anchor'], r['context'], gmap)
                       for _, r in df.iterrows()]
        else:
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

# ── EMA ───────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.model  = model
        self.decay  = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().cpu()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (self.decay * self.shadow[name]
                                     + (1 - self.decay) * param.data.cpu())

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ── AWP (Adversarial Weight Perturbation) ────────────────────
# attack은 gradient '방향'만 사용(norm 정규화)하므로 GradScaler의 scale과 무관.
class AWP:
    def __init__(self, model, adv_lr=1e-4, adv_eps=1e-2):
        self.model   = model
        self.adv_lr  = adv_lr
        self.adv_eps = adv_eps
        self.backup  = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm_grad = torch.norm(param.grad)
                norm_data = torch.norm(param.data)
                if norm_grad != 0 and not torch.isnan(norm_grad):
                    r_at = self.adv_lr * param.grad / (norm_grad + 1e-8) * (norm_data + 1e-8)
                    param.data.add_(r_at)
                    param.data = torch.clamp(
                        param.data,
                        self.backup[name] - self.adv_eps,
                        self.backup[name] + self.adv_eps,
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# ── 모델 (BI-LSTM header) ─────────────────────────────────────
class PatentModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.gradient_checkpointing_enable()
        hidden        = self.backbone.config.hidden_size   # 1024
        self.lstm     = nn.LSTM(hidden, 512, batch_first=True, bidirectional=True)
        self.dropout  = nn.Dropout(0.1)
        self.regressor = nn.Linear(512 * 2, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out      = self.backbone(**kwargs)
        seq_out  = out.last_hidden_state                   # (B, L, 1024)
        lstm_out, _ = self.lstm(seq_out)                   # (B, L, 1024)
        mask     = attention_mask.unsqueeze(-1).float()
        pooled   = (lstm_out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.sigmoid(self.regressor(self.dropout(pooled))).squeeze(-1)

# ── 학습 함수 (EMA + 경량 step ckpt) ───────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler, ema, awp,
                    epoch, fold, best_pearson, start_step=0):
    model.train()
    criterion      = nn.MSELoss()
    step_ckpt_path = f"{CFG['ckpt_dir']}/fold{fold}_step.pt"

    pre_iter_rng = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }

    loader_iter = iter(loader)
    if start_step > 0:
        print(f"  {start_step} steps 건너뜀...", flush=True)
        for _ in range(start_step):
            next(loader_iter)

    total_loss = 0
    optimizer.zero_grad()

    for step, (batch, labels) in enumerate(loader_iter, start=start_step):
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
            # AWP: awp_start_epoch부터 adversarial weight perturbation
            if epoch >= CFG["awp_start_epoch"]:
                awp.attack()
                with torch.cuda.amp.autocast():
                    preds_adv = model(input_ids, attention_mask, token_type_ids)
                    loss_adv  = criterion(preds_adv, labels) / CFG["grad_accum"]
                scaler.scale(loss_adv).backward()   # adversarial grad 누적
                awp.restore()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            ema.update()

        total_loss += loss.item() * CFG["grad_accum"]
        if step % 100 == 0:
            print(f"  step {step}/{len(loader)}  loss={total_loss/(step-start_step+1):.4f}", flush=True)

        if (step + 1) % 2000 == 0:
            torch.save({
                "epoch"        : epoch,
                "step"         : step + 1,
                "model"        : model.state_dict(),
                "best_pearson" : best_pearson,
                "pre_iter_rng" : pre_iter_rng,
            }, step_ckpt_path)

    if os.path.exists(step_ckpt_path):
        os.remove(step_ckpt_path)

    return total_loss / max(len(loader) - start_step, 1)

@torch.no_grad()
def predict(model, loader, ema=None):
    if ema is not None:
        ema.apply_shadow()
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
    if ema is not None:
        ema.restore()
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
train["label"] = (train["score"] * 4).round().astype(int)
skf         = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
fold_splits = list(skf.split(train, train["label"]))

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))

all_data   = pd.concat([train[['anchor','context','target']],
                         test[['anchor','context','target']]], ignore_index=True)
test_gmap  = build_group_map(all_data)
test_ds    = PatentDataset(test, is_test=True, gmap=test_gmap)
test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
                         num_workers=0, pin_memory=False)

progress        = load_progress()
completed_folds = progress.get("completed_folds", {})
n_folds_used    = 0

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
        print(f"Fold {fold} 로드 (pearson={fold_info['pearson']:.4f})", flush=True)

for fold in CFG["train_folds"]:
    if str(fold) in completed_folds:
        print(f"Fold {fold} 이미 완료, 건너뜀", flush=True)
        continue

    tr_idx, val_idx = fold_splits[fold]
    print(f"\n{'='*40}\n FOLD {fold}  (train={len(tr_idx)}, val={len(val_idx)})\n{'='*40}", flush=True)

    tr_df   = train.iloc[tr_idx]
    val_df  = train.iloc[val_idx]
    tr_gmap = build_group_map(tr_df)

    tr_ds      = PatentDataset(tr_df,  augment=True, gmap=tr_gmap)
    val_ds     = PatentDataset(val_df, gmap=tr_gmap)
    tr_loader  = DataLoader(tr_ds,  batch_size=CFG["batch_size"],     shuffle=True,
                            num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
                            num_workers=0, pin_memory=False)

    model     = PatentModel(CFG["model_name"]).to(CFG["device"])
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(),  'lr': CFG['lr']},
        {'params': list(model.lstm.parameters()) + list(model.regressor.parameters()),
         'lr': CFG['head_lr']},
    ], weight_decay=0.01)
    total_steps  = len(tr_loader) // CFG["grad_accum"] * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler()
    ema          = EMA(model, decay=CFG["ema_decay"])
    awp          = AWP(model, adv_lr=CFG["awp_lr"], adv_eps=CFG["awp_eps"])

    start_epoch     = 0
    best_pearson    = -1.0
    epoch_ckpt_path = f"{CFG['ckpt_dir']}/fold{fold}_latest.pt"
    best_ckpt_path  = f"{CFG['ckpt_dir']}/fold{fold}_best.pt"

    step_ckpt_path  = f"{CFG['ckpt_dir']}/fold{fold}_step.pt"
    start_step      = 0

    if os.path.exists(step_ckpt_path):
        ckpt = torch.load(step_ckpt_path, map_location=CFG["device"], weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch  = ckpt["epoch"]
        start_step   = ckpt["step"]
        best_pearson = ckpt["best_pearson"]
        rng = ckpt["pre_iter_rng"]
        torch.set_rng_state(rng['torch'])
        np.random.set_state(rng['numpy'])
        random.setstate(rng['python'])
        fast_steps = start_epoch * (len(tr_loader) // CFG["grad_accum"])
        for _ in range(fast_steps):
            scheduler.step()
        print(f"  Step ckpt: Epoch {start_epoch+1} Step {start_step}부터 재시작", flush=True)
    elif os.path.exists(epoch_ckpt_path):
        ckpt = torch.load(epoch_ckpt_path, map_location=CFG["device"], weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.shadow   = ckpt.get("ema_shadow", {})
        start_epoch  = ckpt["epoch"] + 1
        best_pearson = ckpt["best_pearson"]
        fast_steps = start_epoch * (len(tr_loader) // CFG["grad_accum"])
        for _ in range(fast_steps):
            scheduler.step()
        print(f"  Epoch ckpt: Epoch {start_epoch+1}부터 재시작 (best={best_pearson:.4f})", flush=True)

    for epoch in range(start_epoch, CFG["epochs"]):
        print(f"\n[Fold {fold} | Epoch {epoch+1}/{CFG['epochs']}]", flush=True)
        ep_start_step = start_step if epoch == start_epoch else 0
        train_loss = train_one_epoch(
            model, tr_loader, optimizer, scheduler, scaler, ema, awp,
            epoch=epoch, fold=fold, best_pearson=best_pearson,
            start_step=ep_start_step,
        )

        val_preds  = predict(model, val_loader, ema=ema)
        val_labels = val_df["score"].values
        pearson    = stats.pearsonr(val_labels, val_preds)[0]
        print(f"  train_loss={train_loss:.4f}  val_pearson={pearson:.4f}", flush=True)

        if pearson > best_pearson:
            best_pearson = pearson
            ema.apply_shadow()
            torch.save(model.state_dict(), best_ckpt_path)
            ema.restore()
            print(f"  → Best 저장 (pearson={best_pearson:.4f})", flush=True)

        torch.save({
            "epoch"       : epoch,
            "model"       : model.state_dict(),
            "ema_shadow"  : ema.shadow,
            "best_pearson": best_pearson,
        }, epoch_ckpt_path)

    model.load_state_dict(torch.load(best_ckpt_path, map_location=CFG["device"], weights_only=False))
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

    del model, optimizer, scheduler, scaler, ema, awp, tr_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nFold {fold} 완료. Best pearson={best_pearson:.4f}", flush=True)

# ── OOF 점수 ─────────────────────────────────────────────────
used_idx    = np.concatenate([fold_splits[f][1] for f in CFG["train_folds"]
                               if str(f) in completed_folds])
oof_pearson = stats.pearsonr(train.iloc[used_idx]["score"].values, oof_preds[used_idx])[0]
print(f"\nOOF Pearson: {oof_pearson:.4f}")

# ── 제출 파일 ─────────────────────────────────────────────────
sub["score"] = test_preds / n_folds_used
sub["score"] = sub["score"].clip(0, 1)
sub.to_csv("predictions.csv", index=False)
print("predictions.csv 저장 완료!")
print(sub.head())
