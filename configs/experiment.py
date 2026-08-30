"""Study-wide constants for comparable model-selection experiments."""

PROTOCOL_VERSION = "2.0"

CV_SPLITS = 3
CV_RANDOM_STATE = 10
PARAMETER_RANDOM_STATE = 10

HALVING_FACTOR = 3
HALVING_MIN_RESOURCES = 10
HALVING_MAX_RESOURCES = 1000

# The exact schedule produced by HalvingRandomSearchCV with
# n_candidates="exhaust" and the resource bounds above.
HALVING_CANDIDATES = (100, 34, 12, 4, 2)
HALVING_RESOURCES = (10, 30, 90, 270, 810)

THRESHOLD_METRIC = "f1"
BOOTSTRAP_REPLICATES = 1000
CONFIDENCE_LEVEL = 0.95
