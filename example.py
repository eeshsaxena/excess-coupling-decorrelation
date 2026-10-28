"""Worked synthetic example of excess-coupling decorrelation (E-OA).

Builds three "findings" that are over-coupled by a shared latent factor (the
stand-in for a label source that describes a whole-region condition), shows
that E-OA removes the excess coupling, and shows that the reversed dose makes
it worse, which is the falsification control.

Run:  python example.py
"""
import numpy as np

from eoa import decorrelate, excess_coupling_weights


def mean_pairwise_corr(x):
    c = np.corrcoef(x, rowvar=False)
    k = c.shape[0]
    off = ~np.eye(k, dtype=bool)
    return float(c[off].mean())


def main():
    rng = np.random.default_rng(0)
    n = 4000

    # A shared "whole-region" factor plus a compartment-specific part. The
    # shared factor is what a report-style label source injects into all three.
    shared = rng.normal(size=n)
    specific = rng.normal(size=(n, 3))
    source_scores = 0.9 * shared[:, None] + 0.6 * specific  # over-coupled

    # The true grader couples the three only weakly (mostly specific signal).
    grader = 0.1 * shared[:, None] + 1.0 * specific

    corr_source = np.corrcoef(source_scores, rowvar=False)
    corr_grader = np.corrcoef(grader, rowvar=False)
    w = excess_coupling_weights(corr_source, corr_grader)

    # Embed the three findings as columns 2, 5, 7 of a wider output to show
    # that the other columns are left untouched.
    m = 10
    scores = rng.normal(size=(n, m))
    group = [2, 5, 7]
    scores[:, group] = source_scores

    fixed = decorrelate(scores, group, w, dose=0.2)
    reversed_ = decorrelate(scores, group, w, dose=-0.1)

    print("mean pairwise coupling of the three findings")
    print(f"  source scores        : {mean_pairwise_corr(scores[:, group]):+.3f}")
    print(f"  after E-OA (a=+0.2)  : {mean_pairwise_corr(fixed[:, group]):+.3f}")
    print(f"  reversed   (a=-0.1)  : {mean_pairwise_corr(reversed_[:, group]):+.3f}")

    untouched = np.array_equal(scores[:, [0, 1, 3, 4, 6, 8, 9]],
                               fixed[:, [0, 1, 3, 4, 6, 8, 9]])
    print(f"\nnon-group columns unchanged: {untouched}")


if __name__ == "__main__":
    main()
