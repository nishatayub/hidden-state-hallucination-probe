"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    train_test_split,
)


_CONTEXT_PATTERN = re.compile(
    r"single brief but complete sentence\.\s*(.*?)\s*"
    r"Note that your answer should be strictly based",
    re.S,
)


def _extract_context_group(prompt: str) -> str:
    # 1. Try primary specific pattern (context between instruction and constraints)
    match = _CONTEXT_PATTERN.search(prompt)
    if match:
        return match.group(1).strip()
    
    # 2. Fallback: Take the core content between the user tag and the final question
    # This works for most ChatML formatted prompts even if the wording changes.
    try:
        if "<|im_start|>user" in prompt:
            core = prompt.split("<|im_start|>user")[-1]
            if "Here is the question:" in core:
                return core.split("Here is the question:")[0].strip()
            if "Note that your answer" in core:
                return core.split("Note that your answer")[0].strip()
            return core[:200] # Take a prefix of the user message as group
    except Exception:
        pass

    return prompt


def _group_labels(idx: np.ndarray, groups: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = groups[idx]
    group_targets = []
    for group in unique_groups:
        group_idx = idx[groups[idx] == group]
        labels = y[group_idx]
        group_targets.append(int(labels.mean() >= 0.5))
    return unique_groups, np.asarray(group_targets, dtype=int)


def _group_split(
    idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    holdout_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=24,
        test_size=holdout_size,
        random_state=random_state,
    )

    best_split: tuple[np.ndarray, np.ndarray] | None = None
    full_pos_rate = float(y[idx].mean())
    best_score = float("inf")

    for split_id, (fit_loc, holdout_loc) in enumerate(splitter.split(idx, y[idx], groups[idx])):
        fit_idx = idx[fit_loc]
        holdout_idx = idx[holdout_loc]
        if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[holdout_idx])) < 2:
            continue

        pos_rate_gap = abs(float(y[holdout_idx].mean()) - full_pos_rate)
        size_gap = abs(len(holdout_idx) / len(idx) - holdout_size)
        score = pos_rate_gap + 0.5 * size_gap + split_id * 1e-6
        if score < best_score:
            best_score = score
            best_split = (fit_idx, holdout_idx)

    if best_split is None:
        return train_test_split(
            idx,
            test_size=holdout_size,
            random_state=random_state,
            stratify=y[idx],
        )

    return best_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The default strategy performs a single stratified random split preserving
    the class ratio in each subset.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    idx = np.arange(len(y), dtype=int)

    if df is None or "prompt" not in df:
        idx_train_val, idx_test = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        relative_val = val_size / (1.0 - test_size)
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=relative_val,
            random_state=random_state,
            stratify=y[idx_train_val],
        )
        return [(idx_train, idx_val, idx_test)]

    groups = df["prompt"].map(_extract_context_group).to_numpy()
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    n_group_splits = min(n_splits, len(unique_groups))
    if n_group_splits < 2:
        idx_train_val, idx_test = _group_split(
            idx,
            y,
            groups,
            holdout_size=test_size,
            random_state=random_state,
        )

        relative_val = val_size / (1.0 - test_size)
        idx_train, idx_val = _group_split(
            idx_train_val,
            y,
            groups,
            holdout_size=relative_val,
            random_state=random_state + 1,
        )

        idx_train = np.sort(idx_train)
        idx_val = np.sort(idx_val)
        idx_test = np.sort(idx_test)
        return [(idx_train, idx_val, idx_test)]

    group_labels = []
    for group in unique_groups:
        group_mask = groups == group
        group_labels.append(int(y[group_mask].mean() >= 0.5))
    group_labels = np.asarray(group_labels, dtype=int)

    group_cv = StratifiedGroupKFold(
        n_splits=n_group_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for fold_id, (group_train_loc, group_test_loc) in enumerate(
        group_cv.split(unique_groups, group_labels, unique_groups)
    ):
        train_groups = set(unique_groups[group_train_loc])
        test_groups = set(unique_groups[group_test_loc])

        idx_train_full = np.flatnonzero(np.isin(groups, list(train_groups)))
        idx_test = np.flatnonzero(np.isin(groups, list(test_groups)))

        if len(np.unique(y[idx_train_full])) < 2 or len(np.unique(y[idx_test])) < 2:
            continue

        relative_val = val_size / (1.0 - test_size)
        idx_train, idx_val = _group_split(
            idx_train_full,
            y,
            groups,
            holdout_size=relative_val,
            random_state=random_state + 100 + fold_id,
        )

        folds.append((np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)))

    if folds:
        return folds

    idx_train_val, idx_test = _group_split(
        idx,
        y,
        groups,
        holdout_size=test_size,
        random_state=random_state,
    )
    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val = _group_split(
        idx_train_val,
        y,
        groups,
        holdout_size=relative_val,
        random_state=random_state + 1,
    )
    return [(np.sort(idx_train), np.sort(idx_val), np.sort(idx_test))]
