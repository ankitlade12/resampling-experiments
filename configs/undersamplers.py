from imblearn.under_sampling import (
    RandomUnderSampler,
    CondensedNearestNeighbour,
    TomekLinks,
    OneSidedSelection,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    NeighbourhoodCleaningRule,
    NearMiss,
)

# undersamplers are set up with the configurations
# described in the original articles.

rus = RandomUnderSampler(
    sampling_strategy="auto",
    random_state=0,
    replacement=False,
)

cnn = CondensedNearestNeighbour(
    sampling_strategy="auto",
    random_state=0,
    n_neighbors=1,
    n_jobs=-1,
)

enn = EditedNearestNeighbours(
    sampling_strategy="auto",
    n_neighbors=3,
    kind_sel="all",
)

renn = RepeatedEditedNearestNeighbours(
    sampling_strategy="auto",
    n_neighbors=3,
    kind_sel="all",
    max_iter=100,
)

allknn = AllKNN(
    sampling_strategy="auto",
    n_neighbors=5,
    kind_sel="all",
)

tomek = TomekLinks(
    sampling_strategy="auto",
)

oss = OneSidedSelection(
    sampling_strategy="auto",
    random_state=0,
    n_neighbors=1,
)

ncr = NeighbourhoodCleaningRule(
    sampling_strategy="auto",
    n_neighbors=3,
    threshold_cleaning=0.5,
)

nm1 = NearMiss(
    sampling_strategy="auto",
    version=1,
    n_neighbors=3,
)

nm2 = NearMiss(
    sampling_strategy="auto",
    version=2,
    n_neighbors=3,
)
