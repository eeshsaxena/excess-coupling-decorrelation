import numpy as np

from eoa import percentile_rank, excess_coupling_weights, decorrelate


def _mean_pairwise_corr(x):
    c = np.corrcoef(x, rowvar=False)
    off = ~np.eye(c.shape[0], dtype=bool)
    return float(c[off].mean())


def _over_coupled(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=n)
    specific = rng.normal(size=(n, 3))
    source = 0.9 * shared[:, None] + 0.6 * specific
    grader = 0.1 * shared[:, None] + 1.0 * specific
    w = excess_coupling_weights(np.corrcoef(source, rowvar=False),
                                np.corrcoef(grader, rowvar=False))
    return source, w


def test_percentile_rank_range_and_order():
    x = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    r = percentile_rank(x)
    assert r.min() == 0.0 and r.max() == 1.0
    # order is preserved
    assert np.array_equal(np.argsort(x), np.argsort(r))


def test_weights_average_to_one():
    source, w = _over_coupled()
    off = ~np.eye(3, dtype=bool)
    assert abs(w[off].mean() - 1.0) < 1e-9
    assert np.allclose(np.diag(w), 0.0)


def test_non_group_columns_unchanged():
    rng = np.random.default_rng(1)
    source, w = _over_coupled()
    scores = rng.normal(size=(source.shape[0], 8))
    group = [1, 4, 6]
    scores[:, group] = source
    out = decorrelate(scores, group, w, dose=0.2)
    others = [c for c in range(8) if c not in group]
    assert np.array_equal(scores[:, others], out[:, others])


def test_positive_dose_reduces_coupling_reversed_increases_it():
    source, w = _over_coupled()
    scores = source.copy()
    group = [0, 1, 2]
    base = _mean_pairwise_corr(scores)
    fixed = _mean_pairwise_corr(decorrelate(scores, group, w, dose=0.2)[:, group])
    reversed_ = _mean_pairwise_corr(decorrelate(scores, group, w, dose=-0.1)[:, group])
    assert fixed < base          # E-OA removes excess coupling
    assert reversed_ > base      # falsification control: reversing it hurts


def test_shape_preserved():
    rng = np.random.default_rng(2)
    source, w = _over_coupled()
    scores = rng.normal(size=(source.shape[0], 5))
    scores[:, [0, 2, 4]] = source
    out = decorrelate(scores, [0, 2, 4], w, dose=0.2)
    assert out.shape == scores.shape
