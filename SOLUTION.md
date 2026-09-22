# SOLUTION

## Reproducibility

### Environment

- Python 3.11
- macOS with Apple Silicon, run on **CPU** — the MPS (Metal) backend was
  tried first but caused a native segfault inside Apple's GPU driver during
  feature extraction, and later a near-total stall on a longer run even
  after that crash was fixed. See `EXPERIMENTS.md`, issue 1, for the
  root cause and fix. `solution.py` still auto-detects CUDA if available;
  CPU is the fallback verified stable end-to-end on this machine.
- Dependencies are listed in `requirements.txt`
- Full-dataset extraction + evaluation takes roughly 90 minutes on CPU on
  this machine (`extract_time_s` in `results.json`).

### Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solution.py
```

Running `python solution.py` produces:

- `results.json`
- `predictions.csv`

### Important implementation details

- The solution keeps the fixed infrastructure untouched and only modifies:
  - `aggregation.py`
  - `probe.py`
  - `splitting.py`
- `solution.py` is used exactly as provided for end-to-end execution, with the geometric feature flag enabled.
- Hidden states are extracted from `prompt + response` using `Qwen/Qwen2.5-0.5B`.
- The evaluation now uses grouped cross-validation based on repeated context passages extracted from the prompt text.

## Results

Full 689-row dataset, 5-fold grouped cross-validation, averaged:

| | Accuracy | F1 | AUROC |
|---|---|---|---|
| Majority-class baseline | 70.13% | 82.43% | N/A |
| Probe (test split) | 69.94% | 81.59% | **64.81%** |

The probe does not consistently beat the majority-class baseline on raw
accuracy across folds (wins in 1 of 5, ties in 1, trails narrowly in 3;
the average 0.19-point gap is within noise for ~140 test rows per fold).
It does show consistent, above-chance discriminative ability: test AUROC
falls between 0.60 and 0.71 in every one of the 5 folds. Full per-fold
numbers are in `results.json`; the debugging and ablation work behind these
numbers — including three ideas that were tested and did not help — is in
`EXPERIMENTS.md`.

## Final Solution Description

### Summary

The final solution focuses on methodological robustness rather than only optimizing a single split score. The main changes are:

1. leakage-aware grouped evaluation
2. response-focused multi-layer hidden-state aggregation
3. a lightweight linear probe with PCA for small-data stability

### 1. Grouped splitting in `splitting.py`

The dataset contains repeated context passages paired with different questions. A naive random split can place closely related contexts in both training and test partitions, which risks optimistic evaluation.

To address this, I:

1. extracted the context passage from each prompt
2. used the extracted context as a group identifier
3. replaced the single row-level split with `StratifiedGroupKFold`
4. created a grouped validation split inside each training fold for threshold tuning

This makes the reported metrics stricter and more trustworthy because the probe must generalize to unseen contexts rather than memorizing repeated passages.

### 2. Hidden-state aggregation in `aggregation.py`

The default baseline used only the last real token from the final transformer layer. I replaced this with a response-focused aggregation strategy.

The final aggregation pipeline:

1. selects the top transformer layers rather than only the final layer
2. focuses on the tail of the sequence, which approximates the assistant response region in the concatenated `prompt + response`
3. pools hidden states with a mix of:
   - mean pooling
   - max pooling
   - last-token representation

This gives a more stable representation of answer behavior than a single-token feature.

### 3. Geometric features in `aggregation.py`

I also added compact geometric and statistical features derived from the response-side hidden states:

1. mean activation norm per selected layer
2. standard deviation of activation norms per selected layer
3. cosine similarity between adjacent layer means
4. cosine similarity between the first and last response token in the final selected layer
5. response token count within the selected tail window

These features were designed to capture representation drift and response-shape signals without exploding feature dimensionality.

### 4. Probe design in `probe.py`

Because the labeled dataset is small, I prioritized a simpler probe over a larger neural network.

The final probe uses:

1. `StandardScaler`
2. optional `PCA` compression
3. a regularized linear layer trained with `BCEWithLogitsLoss`
4. class imbalance handling through a softened `pos_weight` (square root of
   the inverse-frequency ratio, rather than the full ratio — see
   `EXPERIMENTS.md` for why the full ratio fights the accuracy objective)
5. internal early stopping on a held-out training subset
6. validation-based threshold tuning, optimized directly for **accuracy**
   (the competition's primary metric) rather than F1

This is more defensible for a small dataset than a deeper MLP and is less likely to overfit under grouped evaluation.

## Why I Made These Choices

The most important design decision was to improve evaluation integrity first.

1. repeated contexts made leakage a real concern
2. grouped evaluation better matches the actual goal of generalizing across unseen contexts
3. once evaluation was stricter, simpler models became more attractive than higher-capacity probes

The aggregation changes were motivated by the fact that hallucination is usually a response-level phenomenon rather than a single-token phenomenon. Pooling across the response-side hidden states captures richer structure than the original last-token baseline.

## What Contributed Most

The most meaningful contribution was the grouped evaluation setup.

Even if it makes the reported metrics lower than a naive random split, it produces results that are more credible and better aligned with real generalization. This was the main methodological improvement in the repository.

The second most important contribution was replacing the last-token baseline with response-focused multi-layer aggregation.

## Experiments And Failed Attempts

### 1. Single random stratified split

The original approach used a standard row-level stratified split. I discarded this as the main evaluation because repeated contexts in the dataset could leak across train and test.

### 2. Deeper MLP probe

I initially upgraded the baseline MLP with dropout, weight decay, PCA, and early stopping. While this was a reasonable improvement over the original probe, it still felt too flexible for the dataset size and less methodologically convincing under grouped evaluation.

I replaced it with a simpler linear probe to prioritize stability and interpretability.

### 3. Geometric features as a standalone novelty

I added geometric/statistical features, but I did not rely on them alone as the core contribution. On a small dataset, elaborate feature engineering can look more sophisticated than it is, so I kept them as a compact extension to a stronger aggregation backbone.

### 4. Threshold tuning edge cases

Threshold tuning on an imbalanced dataset can collapse toward degenerate low thresholds. I restricted the threshold search range to avoid obviously pathological settings.

### 5. F1-optimized threshold instead of accuracy-optimized

The threshold tuner originally maximized F1 on the validation split. Since
the competition's primary ranking metric is accuracy, and this dataset is
imbalanced (~70% hallucinated), an F1-optimal threshold is not the same
point as an accuracy-optimal one. Switched the tuning objective to
`accuracy_score` directly. (Full detail, including a reproducibility bug
this surfaced, in `EXPERIMENTS.md`, issue 3.)

### 6. Minimal last-layer, last-token aggregation

Tested whether the multi-layer mean/max/last-token pooling was adding real
signal or just noise/dimensionality, by comparing against the simplest
possible aggregation (final layer, last token only) on a fast subsample.
The minimal version scored AUROC 0.45 — below random chance — while the
multi-layer version scored 0.60-0.65 on the same data. This confirmed the
multi-layer aggregation is load-bearing and was kept.

### 7. Full inverse-frequency class weighting

The standard `pos_weight = n_neg / n_pos` fully balances the training
loss 50/50 between classes. That's the right target for balanced accuracy,
but under this dataset's imbalance it works against the raw-accuracy
objective, which naturally rewards leaning toward the majority class when
signal is weak. Tried softening the correction to `sqrt(n_neg / n_pos)`;
this did not produce a measurable change on the same subsample test
(66.77% vs. 67.43% test accuracy, within run-to-run noise), so the gap to
baseline is better explained by the strength of the underlying signal than
by the loss-weighting scheme. Kept the softened version anyway as the more
principled choice given it doesn't fight the eval metric even if the
subsample test couldn't detect a difference.

## Final Notes

This solution is intentionally framed around trustworthy evaluation and a compact probe design. The project goal is not only to increase a single split score, but to build a hallucination detector whose reported performance is more believable under repeated-context data.
