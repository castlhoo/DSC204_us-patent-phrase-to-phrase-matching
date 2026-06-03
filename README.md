# US Patent Phrase-to-Phrase Matching

A semantic-similarity model that scores how closely a patent **anchor** phrase matches a **target** phrase within a given technical (CPC) context. The solution fine-tunes **DeBERTa-v3-large** as a regression model and ensembles 5 folds.

> **Final result: 0.87467 Pearson, Gold medal, Top 0.6%, rank 11 / 1889** (up from a 0.85821 baseline at rank 352).

---

## 1. Task

Given an `anchor`, a `target`, and a `context` (a CPC classification code), predict how similar the two phrases are, as a score in `[0, 1]`. The score is meant to reflect similarity *within that patent context*, so the same pair can mean different things under different CPC codes.

**Example row**

| anchor | target | context | score |
|---|---|---|---|
| "abatement" | "eliminating process" | A47 | 0.50 |
| "abatement" | "forest region" | A47 | 0.00 |

**Evaluation metric:** Pearson correlation between predicted and true scores.

| | |
|---|---|
| Train rows | 32,825 |
| Test rows | 3,648 |
| Score labels | 0.0 / 0.25 / 0.5 / 0.75 / 1.0 |

---

## 2. Repository structure

| File | Purpose |
|---|---|
| `deberta_finetune.py` | Main training script. Runs the full 5-fold pipeline and writes `predictions.csv`. |
| `baseline.py` | TF-IDF + cosine-similarity baseline (sanity check). |
| `colab_run.ipynb` | Google Colab runner: pins the right library versions, patches data/checkpoint paths to Drive, and supports resumable training across disconnects. |
| `predictions.csv` | Final 5-fold ensemble predictions (submission file). |

---

## 3. Approach (baseline to final model)

The solution was built in stages, validating each before adding the next.

### Stage 1: TF-IDF baseline (`baseline.py`)
A character-n-gram TF-IDF vectorizer with cosine similarity between the (context + anchor) and (context + target) strings. This was only a sanity check on the data flow and metric; it reached **Train Pearson ≈ 0.44**. Useful to confirm the framing, but far too weak to compete.

### Stage 2: DeBERTa-v3-large regression
We switched to fine-tuning **DeBERTa-v3-large** with a regression head and a sigmoid output, trained with **MSE loss**. The input pairs the anchor with the target as two segments. This immediately and dramatically outperformed the TF-IDF baseline.

### Stage 3: Target-groupby (the key feature trick)
The single most impactful input change. For each `(anchor, context)` pair, we gather the **other targets that share the same anchor** and concatenate them into the input:

```
[CLS] context_text anchor [SEP] target; other_target_1; other_target_2; ... [SEP]
```

This gives the model the "field" of related targets, so it can judge how similar a target is *relative to its peers* under the same anchor. Every top Kaggle solution identifies this grouping as the central trick for this competition. The current target is always excluded from its own group, and the group order is shuffled each epoch.

### Stage 4: Architecture and training stabilizers
On top of the backbone we added:

| Component | What it does | Why it helps |
|---|---|---|
| **BiLSTM head** + mean pooling | a bidirectional LSTM over the token outputs, then masked mean-pooling | builds a richer sequence-level representation than the raw [CLS] token |
| **EMA** (decay 0.999) | keeps an exponential moving average of the weights, used for validation/inference | smooths out late-training noise, giving a more stable and slightly better model |
| **Differential LR** | backbone `2e-5`, head `1e-3` | the pretrained backbone needs gentle updates; the fresh head needs faster learning |
| **Swap augmentation** | also train on (target, anchor) with the same score | similarity is symmetric, so this doubles the data for free |
| **AMP** | autocast + GradScaler mixed precision | faster training and lower memory on GPU |

This full configuration reached a verified **0.858** in a single run on the UCSD **DSMLP** cluster, and became our reference baseline.

### Stage 5: Pushing past the baseline
Two changes on top of the reproduced baseline:

- **`max_length` 128 → 192.** Because target-groupby concatenates several targets, 128 tokens were silently truncating the grouped context. Extending to 192 recovered it (about **+0.006**).
- **AWP (Adversarial Weight Perturbation), starting at epoch 2.** At each optimizer step (after a normal forward/backward), AWP nudges the weights a small amount in the *adversarial* gradient direction, runs a second forward/backward in that perturbed state, and accumulates that gradient before stepping. This forces the model to stay accurate even when its weights are slightly worsened, which improves generalization. It was the **largest single gain** (fold 0: 0.844 → **0.868**, **+0.024**).

Final predictions are a **5-fold StratifiedKFold ensemble**, averaging the test predictions of all five folds.

**Final hyperparameters:** `max_length=192`, `batch_size=4`, `grad_accum=8` (effective batch 32), `epochs=3`, backbone `lr=2e-5`, head `lr=1e-3`, MSE loss, EMA decay `0.999`, AWP from epoch 2.

---

## 4. The main challenge: a hidden environment bug

The verified code scored **0.858 on DSMLP**, but when we moved the **identical code** to Google Colab to train faster on an A100, training **completely collapsed**: validation Pearson stuck at ~0.06 and loss frozen at 0.069, with not one line of model code changed.

**How we (eventually) found it.** We first burned a lot of time rewriting components on guesses (EMA, batch size, TF32, the BiLSTM head, gradient checkpointing, AMP), all of which were wrong. The turning point was switching from guessing to **measurement**: we added a probe printing the backbone gradient norm and saw that **gradients were flowing but the model still was not learning**. That isolated the problem to the *runtime environment* rather than the code. We then logged into the DSMLP cluster directly and read its exact library versions (`transformers 4.44.2`, `torch 2.5.1`). Colab was running a much newer `transformers`, which loads DeBERTa-v3 differently and silently breaks training. **Pinning `transformers==4.44.2` reproduced 0.858 instantly.**

A few secondary bugs surfaced from the newer PyTorch on Colab and had to be fixed so training could survive the A100's frequent disconnects:

| Symptom | Root cause | Fix |
|---|---|---|
| Training collapse (val ~0.06) | newer `transformers` loads DeBERTa-v3 differently | **pin `transformers==4.44.2`** |
| Checkpoint resume crash | PyTorch 2.6 default `weights_only=True` rejects the numpy RNG state | `weights_only=False` |
| RNG restore `TypeError` | `map_location=cuda` moved the RNG state to GPU | `.cpu().to(torch.uint8)` |
| Best checkpoint disappeared | the cleanup cell deleted in-progress fold checkpoints | fall back to the EMA model when best is missing |

**Lesson:** environment reproducibility (pinning library versions) was as decisive as the model. The very same code scored 0.06 vs 0.87 depending only on the `transformers` version.

---

## 5. What helped vs. what did not

| Change | Effect |
|---|---|
| `max_length` 128 → 192 | grouped targets no longer truncated (about +0.006) |
| **AWP** (epoch 2+) | fold 0: 0.844 → **0.868** (+0.024), the biggest single gain |
| Pearson-correlation loss | **reverted.** At init the sigmoid outputs are near-constant, so the prediction std ≈ 0 and the gradient explodes (loss went up, 0.90 to 0.97) |
| FGM (embedding-only perturbation) | **reverted.** Our integration zeroed the original gradient before the adversarial pass, so it cancelled real learning; AWP is the correct replacement |

---

## 6. Results

**Per-fold validation Pearson**

| Fold | 0 | 1 | 2 | 3 | 4 | OOF |
|---|---|---|---|---|---|---|
| Pearson | 0.8680 | 0.8618 | 0.8661 | 0.8670 | 0.8626 | **0.8620** |

**Baseline vs. final**

| Metric | Baseline | Final |
|---|---|---|
| Single-fold Pearson | 0.8374 | 0.8680 |
| 5-fold OOF Pearson | n/a | 0.8620 |
| **Leaderboard score** | 0.85821 | **0.87467** |
| **Rank** | 352 / 1889 (Top 18.6%) | **11 / 1889 (Top 0.6%)** |
| **Medal** | Above Median | **Gold** |

Improvement over the previous best submission: **+0.01646**. For reference, the original Kaggle competition's 1st-place single-model CV was 0.8627, which our individual folds already exceed.

---

## 7. How to run

### Local / DSMLP
```bash
# 1. Place the competition zip at data/us-patent-phrase-to-phrase-matching.zip
# 2. Train (runs all 5 folds, writes predictions.csv)
python deberta_finetune.py

# 3. Submit
aicodinggym mle submit us-patent-phrase-to-phrase-matching -F predictions.csv
```

### Google Colab (A100)
Open `colab_run.ipynb` and run the cells **in order**:

```
Cell 1 (mount Drive) → 2 (check space) → 3 (check data)
→ 4 (download script + patch paths)
→ 4.5 (pin transformers==4.44.2)   <-- required, do not skip
→ 5 (train)
```

Notes:
- **Always run Cell 4.5.** Without pinning `transformers==4.44.2`, training collapses on Colab.
- If the A100 disconnects, rerun **Cell 4 → 4.5 → 5 only**. Checkpoints are stored on Drive, so training resumes from the last saved step.
- **Never run Cell 6 during training.** It deletes in-progress checkpoints; it is only for cleaning up after a fully abandoned run.

---

## 8. Key takeaways

- **Target-groupby is the core trick** for this task: similarity is best judged relative to other targets sharing the same anchor.
- **AWP is a cheap, large generalization win** for transformer fine-tuning, with no architecture changes.
- **Pin your environment.** When a *verified* result breaks after only a platform change, suspect the environment (library versions) before rewriting the model, and measure (e.g. gradient norms) instead of guessing.
