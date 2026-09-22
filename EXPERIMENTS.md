# Experiment Log

This log records the debugging and ablation work done after the initial
solution (`SOLUTION.md`) was drafted but before it was finalized. It exists
so every claim in `SOLUTION.md` is traceable to a specific run, and so a
reviewer (or a future me) can reproduce the reasoning, not just the code.

All ablations below were run on a **150-row subsample** of `data/dataset.csv`
(`sanity_check.py`, not part of the submission) to get fast, directional
answers before committing to the full 689-row run, which takes 30-60+
minutes on this machine. Absolute numbers on the subsample are noisier than
the full-dataset run (test folds as small as ~28-31 rows) — treat them as
*directional*, not final. Final numbers come from `python solution.py` on
the full dataset, recorded in `results.json`.

## Timeline of issues found and fixed

### 1. MPS backend instability (crash + hang)

**Symptom:** `python solution.py` on Apple Silicon (`mps` device) either
segfaulted a few seconds into feature extraction, or silently stalled
(process alive but CPU time not advancing — stuck in uninterruptible sleep)
partway through.

**Root cause:** `aggregation.py`'s `_selected_layers` indexes the hidden-state
tensor with a Python list (`hidden_states[[a, b, c, d]]`, advanced/fancy
indexing) while the tensor still lives on the MPS device. This triggered a
`SIGSEGV` inside Apple's Metal driver (`AGXMetalG14G`), confirmed via
`~/Library/Logs/DiagnosticReports/python3.11-*.ips`. This is a known class of
instability in PyTorch's MPS backend with advanced indexing, not a bug in
our aggregation logic per se.

**Fix:** move each sample's hidden-state tensor to CPU immediately after the
forward pass (`hidden = torch.stack(outputs.hidden_states, dim=1).float().cpu()`
in `solution.py`), before any indexing happens. The forward pass itself still
runs on GPU/MPS when available; only the lightweight aggregation math moved
to CPU.

**Status:** on this machine, even with the fix, MPS extraction degraded
badly over a long run (batches went from ~2s to ~30s, then stalled
completely). All later runs in this log use `device = torch.device("cpu")`
for reliability. `solution.py` still auto-detects CUDA/MPS/CPU for
portability — the CPU fallback is what's actually verified working here.

### 2. Uncommitted changes contradicted the SOLUTION.md narrative

Before any of the ablations below, a review of the uncommitted diff to
`aggregation.py` / `probe.py` found two issues:

- **Feature-dimension blowup.** The pooling in `aggregate()` had grown to
  concatenate mean+max+last-token pooling across *all 4* selected layers
  (12 x 896 = 10,752 dims) on a 479-row training fold — a ~3x increase over
  the previous 3,584-dim baseline, with no accompanying justification.
- **Probe complexity regression.** `_build_network` had been changed from a
  single linear layer to a `Linear -> BatchNorm -> ReLU -> Dropout -> Linear`
  MLP — exactly the architecture `SOLUTION.md`'s "Experiments and Failed
  Attempts" section says was tried and rejected as too flexible for 689
  samples. Code and write-up directly contradicted each other.

**Fix:** reverted the probe to the plain linear layer, and trimmed
`aggregate()` back down: mean-pool across all 4 selected layers (keeps the
"how does the representation drift with depth" signal) but only
max/last-token pool the final layer, not all four. New feature dim: ~5,376
(5,397 with geometric features) instead of 10,752.

### 3. Threshold objective mismatched the competition metric

`fit_hyperparameters` tuned the decision threshold to maximize **F1**, but
the competition's primary ranking metric is **accuracy**. Under this
dataset's class imbalance (~70% label=1 / hallucinated), an F1-optimal
threshold and an accuracy-optimal threshold are not the same point — F1
rewards catching more of the majority class's true positives even at some
accuracy cost.

**Fix:** `fit_hyperparameters` now searches for the threshold that maximizes
`accuracy_score` on the validation split.

**Caveat found while testing this:** the network's `nn.Linear` weights have
no fixed random seed, so the fix appeared to make things *worse* on first
comparison (test accuracy 68.10% -> 63.56%) purely due to run-to-run
initialization noise — confirmed because test AUROC (which threshold choice
cannot affect) also moved between runs. See issue 4.

### 4. No fixed random seed (reproducibility gap)

**Symptom:** rerunning the identical pipeline produced different metrics
each time (test accuracy varied 63.56% / 67.43% / 68.10% across otherwise
identical runs).

**Root cause:** `probe.fit()` builds an `nn.Linear` with PyTorch's default
(unseeded) initialization. Every run starts from different weights.

**Fix:** added `torch.manual_seed(13)` at the start of `fit()`. This doesn't
eliminate all variance (data-loading order, PCA solver internals, etc. can
still vary slightly) but removes the largest source of run-to-run noise and
makes A/B comparisons between code changes actually mean something.

**This matters beyond this repo:** any reported number in `SOLUTION.md`
going forward should be read as "one seeded run," not as a guaranteed
reproduction — full multi-seed averaging (see "Open Questions" below) is the
correct fix and hasn't been done yet due to compute cost on this machine.

## Ablation results (150-row subsample, CPU, seeded from issue 4 onward)

| # | Configuration | Test Acc | Test F1 | Test AUROC | Notes |
|---|---|---|---|---|---|
| 0 | Majority-class baseline | 72.03% | 83.71% | N/A | Always predicts the majority label |
| 1 | Multi-layer aggregation, F1-tuned threshold, **unseeded** | 68.10% | 76.75% | 65.52% | First post-fix run |
| 2 | Multi-layer aggregation, **accuracy**-tuned threshold, unseeded | 63.56% | 73.64% | 60.10% | AUROC moved despite threshold-only change -> confirmed unseeded noise, not a real regression |
| 3 | Multi-layer aggregation, accuracy-tuned threshold, **seeded** | 67.43% | 77.22% | 62.08% | Clean, reproducible baseline for further ablations |
| 4 | **Minimal** aggregation (last layer, last token only), accuracy-tuned, seeded | 68.10% | 80.77% | **45.09%** | AUROC below chance — confirms multi-layer pooling carries real signal; reverted immediately |
| 5 | Multi-layer aggregation (reverted from #4) + **softened pos_weight** (sqrt of inverse-frequency ratio instead of full ratio), accuracy-tuned, seeded | 66.77% | 75.52% | 61.44% | No meaningful change vs #3 (67.43% / 62.08%) — within run-to-run noise. Kept in the code as a small correctness fix (the full ratio was still fighting the accuracy objective in principle) but it is not the lever that closes the gap |

**Reading the table so far:** no configuration has cleared the majority-class
accuracy baseline on this subsample. Configuration #3 (equivalently #5,
since they're statistically indistinguishable) is the best seeded
configuration to date (AUROC ~0.61-0.62, genuinely above chance) and is what
full-dataset runs should be compared against. Configuration #4 is useful
negative evidence: it rules out "the multi-layer complexity is just noise"
as an explanation for the accuracy gap. Configuration #5 rules out "the loss
imbalance weighting is the reason accuracy lags" as the primary
explanation — the gap to baseline is more likely a genuine signal-strength
limitation (AUROC ~0.61) than a training-objective misalignment at this
point.

## Full-dataset run

All ablations above were on a 150-row subsample. The full run (`python
solution.py`, full 689-row dataset, 5-fold grouped CV, CPU — see issue 1 for
why MPS is not used) gives a materially better picture than the subsample
predicted:

| | Accuracy | F1 | AUROC |
|---|---|---|---|
| Majority-class baseline (avg over 5 folds) | 70.13% | 82.43% | N/A |
| Probe, test split (avg over 5 folds) | 69.94% | 81.59% | **64.81%** |

Per-fold accuracy (probe vs. baseline): fold 1 — 70.00% vs 70.00% (tie);
fold 2 — **70.92% vs 66.67%** (probe wins); fold 3 — 68.94% vs 71.97%;
fold 4 — 70.42% vs 71.13%; fold 5 — 69.40% vs 70.90%.

**Reading this honestly:** the probe does not reliably beat the
majority-class baseline on accuracy — it wins outright in 1 of 5 folds,
ties in 1, and trails narrowly (1-3 points) in the other 3. Averaged, it is
statistically indistinguishable from the baseline (a 0.19-point gap on
~140 test rows per fold is noise, not a real effect). What *did* improve
substantially over the subsample estimate is AUROC: 0.648 vs the
subsample's 0.60-0.62. This matches the prediction in the "Open Questions"
section below — more data (and 5 real folds instead of 1) let the
underlying signal show up more clearly, even though it wasn't enough to
consistently move the needle on raw accuracy given this dataset's ~70/30
class imbalance.

**Fair characterization for `SOLUTION.md` and any external write-up:** this
probe demonstrates genuine, above-chance separability in Qwen2.5-0.5B's
hidden states between truthful and hallucinated responses (AUROC 0.65,
consistent across all 5 folds — every single fold's test AUROC is between
0.60 and 0.71, never near or below 0.50). It does not yet consistently
outperform a naive majority-class heuristic on raw accuracy under this
dataset's imbalance. Both of those are true at once, and reporting only the
first without the second would be overstating the result.

## Open questions / not yet done

- **Full-dataset run.** All ablations above are on a 150-row subsample for
  speed. The full 689-row run has ~3x the test-fold size per fold (100 vs
  ~30), which should reduce metric variance substantially and may reveal a
  different ranking between configurations than the subsample shows.
- **Multi-seed averaging.** Issue 4 fixed single-run reproducibility, but a
  defensible final number should average over 3-5 seeds, not one.
- ~~Why `n_folds` collapses to 1 on the full dataset but works as 5 on the
  subsample.~~ **Resolved** — the full run now correctly produces 5 folds.
  The original `n_folds: 1` result predates the `splitting.py` context-group
  fallback added during this debugging session (see README's "What makes
  this more than a script" section); the improved fallback extraction
  appears to have fixed the group collapse.
- **Response-window sensitivity.** Not yet tested: whether hallucination
  signal concentrates in the first few generated tokens (where the model
  commits to a claim) rather than being spread across the whole tail window
  currently used.
