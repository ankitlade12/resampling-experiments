# Research validity protocol (v2)

## Status

Protocol v2 supersedes all model scores, figures, and narrative conclusions
embedded in the existing notebooks. Those outputs were generated under the
legacy v1 workflow and are exploratory only. They cannot be repaired by editing
tables: every affected model must be retrained and every held-out prediction
must be regenerated.

The v1 results are invalid for confirmatory claims because the workflow selected
decision thresholds on test labels, repeatedly optimized variants against the
test set, used only five partial bootstraps, split repeated diabetes encounters
across train and test, and fit some preprocessing steps before splitting.

## Confirmatory question and estimands

The primary question is whether an undersampling method improves an otherwise
matched ensemble on genuinely unseen observations. Comparisons are paired by
dataset, split, estimator family, hyperparameter-search budget, and test rows.

- Primary ranking metric: average precision (AP), because the positive class is
  rare and the metric directly reflects precision-recall performance.
- Secondary threshold-free metric: ROC-AUC.
- Probability-quality metric: Brier loss. For undersampled models, confirmatory
  Brier claims require calibration learned only from original-prevalence
  training folds; raw undersampled probabilities are exploratory.
- Operational metrics: precision, recall, F1, MCC, balanced accuracy, and
  G-mean at one frozen threshold learned from training-only OOF predictions.

Positive paired effects always mean that the candidate is better. Thus AP and
ROC effects are candidate minus reference, while Brier effects are reference
minus candidate.

## Data and split rules

1. Deduplicate and define the cohort before model development. Diabetes records
   ending in death or hospice are excluded because the readmission endpoint is
   not meaningful for those encounters.
2. Hold out 30% once. Ordinary datasets use a stratified row split. Diabetes
   uses a group-aware patient split, with no patient in both partitions.
3. Fit imputers, encoders, scalers, variance/constant-feature filters, samplers,
   and feature selection on training data only. Test data is transform-only.
4. Store dataset/split fingerprints in `run_manifest.json`. A changed split or
   dataset must use a new output directory.
5. Never use test labels for preprocessing, hyperparameter selection, sampler
   selection, probability calibration, decision thresholds, or early stopping.

## Model selection and fairness

Baseline and pre-undersampled ensemble searches use identical shuffled
three-fold partitions, random seeds, scoring, and successive-halving resources:

| Stage | Candidates | Trees |
|---:|---:|---:|
| 1 | 100 | 10 |
| 2 | 34 | 30 |
| 3 | 12 | 90 |
| 4 | 4 | 270 |
| 5 | 2 | 810 |

The final configuration is refit with 810 trees. Its operating threshold is
chosen by maximizing F1 on concatenated OOF training predictions, then frozen.
If sampler choice itself is optimized, it must occur inside a nested training
CV loop; choosing the displayed winner from the final test set is exploratory.

## Evaluation and inference

- Report point estimates on the entire held-out set. Do not average partial test
  subsets and call the result test performance.
- Use at least 1,000 bootstrap replicates for 95% percentile intervals.
- Use a stratified row bootstrap for independent observations. For diabetes,
  resample whole patients so repeated encounters remain correlated.
- Compare models with the same bootstrap draw via
  `paired_bootstrap_difference`; independent model intervals are not a test of
  the difference.
- Pre-specify the comparison family. When multiple p-values are reported within
  a dataset/metric family, apply `holm_adjust`. Confidence intervals and effect
  sizes remain mandatory; a color-coded difference is not significance.
- Report prevalence, sample/class/patient counts, missingness, and the number of
  valid bootstrap replicates with every table.

## Reproducibility and run control

Protocol-v2 training entry points create `run_manifest.json` before reading
resumable artifacts. The manifest binds a run to the full configuration, Git
commit and clean/dirty state, Python/OS/core package versions, and exact
train/test array hashes.

A dirty worktree, changed configuration/code/data, or a directory containing
legacy artifacts without a manifest stops the run. Preserve old artifacts, but
start v2 in a clean directory rather than mixing versions.

## Required rerun sequence

1. Commit protocol code and create a clean environment from `requirements.txt`.
2. Archive legacy model directories and use new empty v2 output directories.
3. Retrain ordinary baselines and special ensembles.
4. Retrain every undersampling and IHT variant with the matched search schedule.
5. Add training-only probability calibration for undersampled models before any
   confirmatory Brier comparison.
6. Generate full-test metrics, intervals, and retained prediction bundles.
7. Run only pre-specified paired comparisons, with multiplicity correction.
8. Rebuild notebooks from cleared outputs and replace all v1 prose with results
   supported by v2 artifacts and manifests.
