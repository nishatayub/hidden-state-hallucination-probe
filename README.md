# Hallucination Detection from LLM Hidden States

A lightweight probe that reads a small language model's internal
representations and predicts whether its own answer is hallucinated —
without needing a second, larger model to check its work.

Built for the SMILES-2026 research application task, using
[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) as the base model.

## The problem

Language models sometimes generate plausible-sounding but factually wrong
answers. The usual way to catch this is expensive: run a second, larger
model as a judge, or check the answer against a knowledge base. This project
asks a cheaper question instead — **does the small model's own hidden state
already "know" it's about to hallucinate?** If truthful and hallucinated
responses activate the model's internals differently in a way a lightweight
classifier can pick up on, you get hallucination detection almost for free,
with no extra inference cost beyond the forward pass you were already doing.

## Approach

1. **Extract hidden states.** For every `prompt + response` pair, run one
   forward pass through Qwen2.5-0.5B and keep the hidden states at every
   layer, for every token (`aggregation.py`, `solution.py`).
2. **Aggregate into a feature vector.** Rather than using only the final
   layer's last token (the common baseline), pool across a spread of layers
   at different depths and across the response-token window, capturing how
   the representation evolves as the model "commits" to its answer.
3. **Classify with a probe.** A `StandardScaler` → `PCA` → single linear
   layer, trained with a class-imbalance-aware loss (`probe.py`). Kept
   deliberately simple — see [Why a linear probe, not an MLP](#why-a-linear-probe-not-an-mlp) below.
4. **Evaluate honestly.** Contexts repeat across rows in this dataset (the
   same passage paired with different questions), so a naive random split
   leaks information between train and test. Splits are grouped by context
   passage instead (`splitting.py`), which produces a stricter but more
   trustworthy estimate of how well the probe generalizes to genuinely
   unseen material.

## What makes this more than a script that runs

Most solutions to a task like this stop at "train a model, report a number."
The things below are what separates that from an actual investigation —
each one is a real finding from debugging this specific pipeline, not a
hypothetical. Full detail, numbers, and reasoning for every item are in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

- **Found and fixed a native crash in the training loop.** Running feature
  extraction on Apple Silicon's MPS backend intermittently segfaulted deep
  inside Apple's own Metal driver. Root-caused it to advanced tensor
  indexing happening while data still lived on the GPU device, and fixed it
  by moving the lightweight aggregation math to CPU right after the forward
  pass — keeping the expensive part (the model forward pass) on GPU while
  avoiding the unstable code path.
- **Caught a metric/objective mismatch.** The decision threshold was
  originally tuned to maximize F1, but the competition's primary ranking
  metric is accuracy — under this dataset's class imbalance, those two
  optimize for different thresholds. Re-tuned the objective to match what's
  actually being scored.
- **Found a reproducibility bug while verifying the fix above.** The probe's
  linear layer had no fixed random seed, so identical code produced
  different metrics on every run — which meant an apparent "improvement"
  from the threshold fix was actually just noise. Pinned the seed, which is
  also just good practice: a result nobody else can reproduce isn't a result.
- **Ran an ablation instead of assuming the fancy version is better.**
  Tested the multi-layer aggregation against the simplest possible baseline
  (last layer, last token only) on the same data. The simple version scored
  *below random chance* (AUROC 0.45) — concrete evidence that the
  multi-layer pooling is carrying real signal, not just adding
  dimensionality for its own sake.
- **Diagnosed a loss/metric tension.** The standard inverse-frequency class
  weighting fully balances the training loss 50/50 between classes — the
  right target if you're optimizing balanced accuracy, but it fights the
  raw-accuracy objective this competition actually scores, since accuracy
  under imbalance naturally rewards leaning toward the majority class when
  the signal is weak. Softened the correction and re-tested; it didn't
  change the outcome, which is itself useful evidence that the accuracy gap
  comes from signal strength, not the training objective (see
  `EXPERIMENTS.md`).

### Why a linear probe, not an MLP

It's tempting to reach for a deeper network, and an early version of this
project did exactly that — a `Linear → BatchNorm → ReLU → Dropout → Linear`
MLP. It was reverted. With 689 labeled examples and a probe that operates on
several-thousand-dimensional pooled hidden states, a higher-capacity model
is easier to overfit and harder to trust under the grouped, leakage-aware
evaluation this project uses. A regularized linear layer on top of PCA is a
smaller hypothesis space that's more defensible given how little labeled
data there is — the point of a probe is to test whether the *representation*
is linearly separable, not to see how much a bigger classifier can squeeze
out of it.

## Results

Final metrics, from `solution.py` on the full 689-row dataset with 5-fold
grouped cross-validation (full numbers in [`results.json`](results.json)):

| | Accuracy | F1 | AUROC |
|---|---|---|---|
| Majority-class baseline | 70.13% | 82.43% | N/A |
| **Probe (this project)** | 69.94% | 81.59% | **64.81%** |

**The honest read:** the probe does not reliably beat the majority-class
baseline on raw accuracy — it wins outright in 1 of 5 folds, ties in 1, and
trails narrowly in the other 3, averaging out to a statistically
insignificant 0.19-point gap. What it does show clearly is genuine,
above-chance separability in the model's hidden states: test AUROC is
between 0.60 and 0.71 in *every single fold*, never near random. That's the
real finding — this small model's internals do encode something about
whether it's about to hallucinate, even though a simple linear probe on
pooled hidden states isn't yet enough to turn that signal into a reliable
accuracy win under this dataset's ~70/30 class imbalance and 689-row size.

Fast diagnostic ablations on a 150-row subsample (used to iterate in
~15 minutes instead of the full run's ~90) are tracked separately in
[`EXPERIMENTS.md`](EXPERIMENTS.md), including cases where an idea *didn't*
work — a probe with no real signal, or a "fix" that turned out to be
initialization noise. Reporting those is part of the point: a result that
only shows the wins isn't a trustworthy one.

## Repository Structure

```
├── data/
│   ├── dataset.csv        # Labelled training data (prompt, response, label)
│   └── test.csv           # Unlabelled competition test set
│
├── solution.py             # Entry point: extraction → aggregation → probe → predictions.csv
├── aggregation.py          # Layer selection, token pooling, geometric features
├── probe.py                # HallucinationProbe — the binary classifier
├── splitting.py             # Grouped, leakage-aware train / val / test splitting
│
├── model.py                # Fixed infra: loads Qwen2.5-0.5B
├── evaluate.py              # Fixed infra: evaluation loop, metrics, JSON output
│
├── SOLUTION.md              # Formal write-up: methodology, reasoning, what was tried and discarded
├── EXPERIMENTS.md           # Debugging log + ablation results with numbers
└── results.json             # Final metrics from the full-dataset run (committed, per submission rules)
```

`predictions.csv` is generated by `python solution.py` but isn't committed —
per the competition's submission format, it's uploaded separately to cloud
storage and linked in the application, not tracked in this repo.

## Quick Start

```bash
git clone <this-repo>
cd SMILES-2026-Hallucination-Detection

python -m venv .venv
source .venv/bin/activate        # Linux / macOS

pip install -r requirements.txt
python solution.py
```

This re-extracts hidden states for the full dataset and test set, trains the
probe under grouped cross-validation, and writes `results.json` and
`predictions.csv`. See `SOLUTION.md` for exact reproducibility details
(environment, runtime, hardware notes).

## Dataset

`data/dataset.csv` contains 689 labelled samples:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Full ChatML-formatted conversation context fed to Qwen |
| `response` | str | The model's generated response |
| `label` | float | `1.0` = hallucinated · `0.0` = truthful |

`data/test.csv` is structured identically but unlabeled — `predictions.csv`
holds this project's predictions for it.
