# US Patent Phrase-to-Phrase Matching

Semantic similarity model that scores how closely a patent **anchor** phrase matches a **target** phrase within a given CPC context. Built by fine-tuning DeBERTa-v3-large as a regression model.

**Final result: 0.87467 (Pearson), Gold medal, Top 0.6%, rank 11 / 1889.**

---

## Task

Given an `anchor`, a `target`, and a `context` (CPC code), predict a similarity score in `[0, 1]`.
Evaluation metric: **Pearson correlation** between predicted and true scores.

| | |
|---|---|
| Train | 32,825 rows |
| Test | 3,648 rows |
| Score labels | 0.0 / 0.25 / 0.5 / 0.75 / 1.0 |

---

## Approach (baseline to final model)

We built up the solution in clear stages, validating each before adding the next.

**Stage 1: TF-IDF baseline (`baseline.py`).**
A character-n-gram TF-IDF vectorizer with cosine similarity between anchor and target. This was only a sanity check on the data pipeline and metric, reaching Train Pearson around 0.44. It confirmed the task framing but was far too weak to be competitive.

**Stage 2: DeBERTa-v3-large regression.**
We switched to fine-tuning DeBERTa-v3-large with a linear regression head and a sigmoid output, trained with MSE loss. This is a strong semantic model and immediately outperformed the TF-IDF baseline by a wide margin.

**Stage 3: Input engineering with target-groupby.**
The single most important feature trick. For each `(anchor, context)` pair, we collect the *other* targets that share the same anchor and concatenate them into the input:
`anchor [SEP] target; other_target_1; other_target_2; ... [SEP] CPC_text`.
This lets the model judge a target's similarity in the context of related targets, which all top Kaggle solutions cite as the key to this competition.

**Stage 4: Architecture and training stabilizers.**
On top of the backbone we added a **BiLSTM head** with mean pooling for a richer sequence representation, **EMA** (exponential moving average of weights) for stable validation/inference, **differential learning rates** (backbone 2e-5, head 1e-3), anchor-target **swap augmentation**, and **AMP** mixed-precision training. This configuration reached **0.858** as a verified single run on the UCSD DSMLP cluster, and became our reference baseline.

**Stage 5: Pushing past the baseline.**
On the reproduced baseline we added two improvements:
- **`max_length` 128 to 192.** Because target-groupby concatenates several targets, 128 tokens were truncating useful context. Extending to 192 recovered it.
- **AWP (Adversarial Weight Perturbation), from epoch 2.** During training it perturbs the weights in the adversarial gradient direction, forcing the model to stay robust to small weight changes. This was the largest single gain.

Final predictions are a **5-fold StratifiedKFold ensemble**, averaging the test predictions of all five folds.

Final hyperparameters: `max_length=192`, `batch_size=4`, `grad_accum=8`, `epochs=3`, backbone `lr=2e-5`, head `lr=1e-3`, MSE loss, EMA decay 0.999.

---

## The main challenge: a hidden environment bug

The verified code scored **0.858 on the DSMLP cluster**, but when moved to Google Colab the **exact same code completely failed to train** (validation Pearson stuck at ~0.06, loss frozen at 0.069), with not a single line of model code changed.

After ruling out many code-level suspects, the real cause turned out to be a **library version mismatch**. The fix was to pin the environment to match DSMLP:

| Symptom | Root cause | Fix |
|---|---|---|
| Training collapse (val ~0.06) | Colab's newer `transformers` loads DeBERTa-v3 differently | **Pin `transformers==4.44.2`** (the DSMLP version) |
| Checkpoint resume crash | PyTorch 2.6 default `weights_only=True` rejects numpy RNG state | `weights_only=False` |
| RNG restore error | `map_location=cuda` moved RNG state to GPU | `.cpu().to(uint8)` |
| Best checkpoint lost | cleanup cell deleted in-progress folds | fallback to EMA model |

**Lesson:** environment reproducibility (pinning library versions) was as decisive as the model itself. The same code scored 0.06 vs 0.87 depending only on the `transformers` version, because a newer release changed how DeBERTa-v3 weights are loaded.

---

## What helped vs. what did not

| Change | Effect |
|---|---|
| `max_length` 128 to 192 | grouped targets no longer truncated (about +0.006) |
| **AWP** (epoch 2+) | fold 0: 0.844 to **0.868** (+0.024), the biggest gain |
| Pearson-correlation loss | reverted (gradient explosion at init, near-constant sigmoid output) |
| FGM (embedding-only perturbation) | reverted (our integration cancelled the original gradient) |

---

## Results

| Metric | Baseline | Final |
|---|---|---|
| Single-fold Pearson | 0.8374 | 0.8680 |
| 5-fold OOF Pearson | n/a | 0.8620 |
| **Leaderboard score** | 0.85821 | **0.87467** |
| **Rank** | 352 / 1889 (Top 18.6%) | **11 / 1889 (Top 0.6%)** |
| **Medal** | Above Median | **Gold** |

Improvement over previous best: **+0.01646**.

---

## Files

| File | Purpose |
|---|---|
| `deberta_finetune.py` | Main training script (5-fold ensemble, produces `predictions.csv`) |
| `baseline.py` | TF-IDF cosine-similarity baseline |
| `colab_run.ipynb` | Colab runner (pins transformers, patches paths, resumable training) |

---

## Run

```bash
# Train (produces predictions.csv via 5-fold ensemble)
python deberta_finetune.py

# Submit
aicodinggym mle submit us-patent-phrase-to-phrase-matching -F predictions.csv
```

> On Colab, pin `transformers==4.44.2` before training (handled by `colab_run.ipynb`, Cell 4.5).
