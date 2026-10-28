"""Excess-coupling decorrelation (E-OA).

A small, model-independent post-processing method that decorrelates a group of
related classifier outputs in percentile-rank space, by the amount by which the
training-label source over-couples them relative to the true grading target.

The method acts only on the output scores: it never touches the network
weights, the architecture or the training procedure, so it works on top of any
model or ensemble. See the README for the method, its boundary conditions and
the validation protocol (including the reversed-dose falsification control).
"""
from __future__ import annotations

import numpy as np

__all__ = ["percentile_rank", "excess_coupling_weights", "decorrelate"]


def _rank_1d(v):
    """Average-rank percentile transform of a 1-D array into [0, 1]."""
    n = v.shape[0]
    if n <= 1:
        return np.zeros_like(v, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    # Give tied values their average rank so the transform is order-only.
    sv = v[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks / (n - 1)


def percentile_rank(x):
    """Map scores to percentile ranks in [0, 1].

    Each value is replaced by its rank among all values divided by ``n - 1``,
    so the smallest becomes 0.0 and the largest 1.0; ties take the average
    rank. A 2-D array is ranked independently per column.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return _rank_1d(x)
    if x.ndim == 2:
        return np.column_stack([_rank_1d(x[:, j]) for j in range(x.shape[1])])
    raise ValueError("percentile_rank expects a 1-D or 2-D array")


def excess_coupling_weights(corr_source, corr_target, normalize=True):
    """Pair weights from the gap between the label source and the true target.

    Parameters
    ----------
    corr_source, corr_target : (k, k) array_like
        Correlation matrices among the ``k`` grouped findings: how strongly the
        label source couples each pair (``corr_source``) and how strongly the
        true grader does (``corr_target``).
    normalize : bool
        If True, scale the off-diagonal weights so they average 1.0, which
        keeps the meaning of ``dose`` in :func:`decorrelate` consistent across
        problems.

    Returns
    -------
    w : (k, k) ndarray
        ``w[t, o]`` is the weight applied to member ``o`` when correcting
        member ``t``. The diagonal is zero.
    """
    s = np.asarray(corr_source, dtype=float)
    g = np.asarray(corr_target, dtype=float)
    if s.shape != g.shape or s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("corr_source and corr_target must be square and equal shape")
    excess = s - g
    np.fill_diagonal(excess, 0.0)
    w = excess.copy()
    if normalize:
        off = ~np.eye(w.shape[0], dtype=bool)
        mean = w[off].mean()
        if mean != 0:
            w = w / mean
        np.fill_diagonal(w, 0.0)
    return w


def decorrelate(scores, group, weights, dose=0.2, rerank=True):
    """Apply E-OA to a group of related columns of ``scores``.

    For each member ``t`` of the group, in percentile-rank space::

        out_t = pct( pct(f_t) - (dose / 2) * sum_{o != t} w[t, o] * pct(f_o) )

    Parameters
    ----------
    scores : (n, m) array_like
        ``n`` studies by ``m`` findings; any real-valued scores.
    group : sequence of int
        Column indices of the over-coupled group (e.g. the three OA
        compartments). Order must match ``weights``.
    weights : (k, k) array_like
        Pair-weight matrix for the ``k = len(group)`` members, from
        :func:`excess_coupling_weights`.
    dose : float
        Scalar ``a``. Positive decorrelates; ``a < 0`` is the falsification
        control and must make results worse if the mismatch is real.
    rerank : bool
        If True, map each corrected column back to percentile ranks.

    Returns
    -------
    out : (n, m) ndarray
        A copy of ``scores`` with only the ``group`` columns changed.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2-D (n_studies, n_findings) array")
    group = list(group)
    k = len(group)
    w = np.asarray(weights, dtype=float)
    if w.shape != (k, k):
        raise ValueError("weights must be (len(group), len(group))")

    out = scores.copy()
    r = percentile_rank(scores[:, group])  # (n, k), rank space
    corrected = np.empty_like(r)
    for ti in range(k):
        adj = np.zeros(r.shape[0])
        for oi in range(k):
            if oi != ti:
                adj = adj + w[ti, oi] * r[:, oi]
        corrected[:, ti] = r[:, ti] - (dose / 2.0) * adj
    if rerank:
        corrected = percentile_rank(corrected)
    for idx, col in enumerate(group):
        out[:, col] = corrected[:, idx]
    return out
