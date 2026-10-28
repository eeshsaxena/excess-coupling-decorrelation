# Excess-coupling decorrelation (E-OA)

A small, **model-independent** post-processing method that decorrelates a group of related
classifier outputs, by the exact amount by which the training-label source *over-couples* them
relative to the true grading target.

E-OA acts only on the output scores. It never touches the network weights, the architecture or
the training procedure, so it works on top of **any** model or ensemble, costs almost nothing (a
few lines of rank arithmetic), and contains no data and no model weights.

It was developed for the three osteoarthritis compartments (medial, lateral, patellofemoral) in a
twelve-finding knee-MRI classifier, where training labels are extracted from free-text radiology
reports (which describe osteoarthritis as one whole-knee property) but the evaluation is graded by
experts who score each compartment separately. The method is general: it applies wherever a label
source couples a group of related outputs more than the true grader does.

## The method

Work in percentile-rank space. Let `pct(.)` map a column of scores to percentile ranks in `[0, 1]`.
For each member `t` of the over-coupled group:

```
out_t = pct( pct(f_t) - (a / 2) * sum_{o != t} w(t, o) * pct(f_o) )
```

- `a` is a single scalar **dose** (e.g. `0.2`). It controls how strong the correction is.
- `w(t, o)` is a **pair weight**, not hand-set. It comes from the measured excess coupling

```
e(t, o) = corr_source(t, o) - corr_grader(t, o)
```

  scaled so the off-diagonal weights average to one.

Intuition: each score is a shared "whole-region" component (common to the group, injected by the
label source) plus a member-specific component. Subtracting a controlled share of the other members
shrinks the shared part by roughly `(1 - a)` while leaving the member-specific part almost untouched,
moving the output toward how the true grader would score it.

## Install

```bash
pip install numpy
pip install -e .          # from a clone of this repository
```

The only dependency is NumPy.

## Quickstart

```python
import numpy as np
from eoa import excess_coupling_weights, decorrelate

# scores: (n_studies, n_findings); here the group is columns 2, 5, 7
group = [2, 5, 7]

# correlation among the group members, as the label source couples them ...
corr_source = np.corrcoef(source_scores, rowvar=False)
# ... and as the true grader couples them (even a small expert set will do)
corr_grader = np.corrcoef(expert_grades, rowvar=False)

w = excess_coupling_weights(corr_source, corr_grader)
out = decorrelate(scores, group, w, dose=0.2)   # only the group columns change
```

Run the worked synthetic demo:

```bash
python example.py
```

## When it helps, and when it does not

E-OA helps **only** where the label source genuinely over-couples a group of outputs relative to the
true target. Applied where no such mismatch exists it does nothing, or it hurts. Always validate:

1. Pick a group of related outputs you suspect the label source over-couples.
2. Estimate how the true grader couples them (even a small independent expert set).
3. Measure the effect on an independent, expert-graded set you did not tune on.
4. Include the **reversed-dose falsification control**: running with `a < 0` must make results
   *worse*. If reversing it does not hurt, the "excess coupling" was not real.

## Tests

```bash
pip install pytest
pytest
```

The tests check the rank transform, that only the group columns change, that the pair weights
average to one, and that a positive dose reduces the group's coupling while the reversed dose
increases it.

## Provenance and citation

This is a reference implementation of the E-OA method from the B.Tech Project-II report
*"Detecting Knee Abnormalities in MRI: A Multi-Model Ensemble with a Label-versus-Grader
Decorrelation Method (E-OA)"* (Eesh Saxena, IIIT Senapati, Manipur, 2026). The repository contains
the method and a synthetic example only; it includes no patient data and no model weights. The
controlled external data used to validate the method in the report is not distributed here.

```bibtex
@misc{saxena2026eoa,
  author = {Eesh Saxena},
  title  = {Excess-coupling decorrelation (E-OA)},
  year   = {2026},
  howpublished = {\url{https://github.com/eeshsaxena/excess-coupling-decorrelation}}
}
```

## License

MIT. See [LICENSE](LICENSE).
