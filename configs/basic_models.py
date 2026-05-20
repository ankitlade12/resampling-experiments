from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

estimator_dict = {
    "logit": Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", LogisticRegression(solver="saga", random_state=10, max_iter=1000000)),
    ]),
    "knn": Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", KNeighborsClassifier()),
    ]),
    "tree": DecisionTreeClassifier(random_state=10),
}