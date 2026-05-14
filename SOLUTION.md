# SOLUTION

## Reproducibility

### Environment

- Python 3.11
- macOS with Apple Silicon (`mps`) was used for development
- Dependencies are listed in `requirements.txt`

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
4. class imbalance handling through `pos_weight`
5. internal early stopping on a held-out training subset
6. validation-based threshold tuning for prediction

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

## Final Notes

This solution is intentionally framed around trustworthy evaluation and a compact probe design. The project goal is not only to increase a single split score, but to build a hallucination detector whose reported performance is more believable under repeated-context data.
