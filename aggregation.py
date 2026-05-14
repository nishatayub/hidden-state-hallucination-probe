"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _real_token_slice(attention_mask: torch.Tensor) -> tuple[int, int]:
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    start = int(real_positions[0].item())
    end = int(real_positions[-1].item()) + 1
    return start, end


def _response_window(attention_mask: torch.Tensor) -> slice:
    start, end = _real_token_slice(attention_mask)
    seq_len = end - start
    tail_len = max(16, seq_len // 3)
    response_start = max(start, end - tail_len)
    return slice(response_start, end)


def _selected_layers(hidden_states: torch.Tensor) -> torch.Tensor:
    # Skip the embedding layer and use the top transformer layers where answer
    # semantics and uncertainty signals tend to be most separable.
    n_transformer_layers = hidden_states.size(0) - 1
    n_take = min(4, n_transformer_layers)
    return hidden_states[-n_take:]


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    layers = _selected_layers(hidden_states)
    response_tokens = layers[:, _response_window(attention_mask), :]

    mean_pool = response_tokens.mean(dim=1)
    max_pool = response_tokens.max(dim=1).values
    last_token = response_tokens[:, -1, :]

    pooled = torch.cat([mean_pool[-2:], max_pool[-1:], last_token[-1:]], dim=0)
    return pooled.reshape(-1)
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    layers = _selected_layers(hidden_states)
    response_tokens = layers[:, _response_window(attention_mask), :]

    token_norms = torch.linalg.vector_norm(response_tokens, dim=-1)
    mean_norm_per_layer = token_norms.mean(dim=1)
    std_norm_per_layer = token_norms.std(dim=1, unbiased=False)

    layer_means = response_tokens.mean(dim=1)
    inter_layer_cos = []
    for i in range(layer_means.size(0) - 1):
        inter_layer_cos.append(
            F.cosine_similarity(layer_means[i], layer_means[i + 1], dim=0)
        )

    trajectory = F.cosine_similarity(
        response_tokens[-1, 0],
        response_tokens[-1, -1],
        dim=0,
    ).unsqueeze(0)
    response_len = torch.tensor([float(response_tokens.size(1))], dtype=hidden_states.dtype)

    features = [
        mean_norm_per_layer,
        std_norm_per_layer,
        torch.stack(inter_layer_cos) if inter_layer_cos else torch.zeros(0),
        trajectory,
        response_len,
    ]
    return torch.cat(features, dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
