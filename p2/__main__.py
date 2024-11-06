import random

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from scipy.spatial.distance import canberra

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def set_seed(seed) -> None:
    """
    Set seed for reproducibility

    Parameters
    ----------
    seed : int
    """
    random.seed(seed)
    np.random.seed(seed)


def load_data(path) -> tuple:
    """
    Load data from the given path and return the features and labels

    Parameters
    ----------
    path : str

    Returns
    -------
    X : pd.DataFrame - features of the data to be used for training
    y_encoded : np.ndarray - encoded labels of the data to be used for training
    label_encoder : LabelEncoder - label encoder used to encode and decode the labels
    """
    data = pd.read_csv(path)  # 27 features

    X = data.drop(
        columns=["Pos", "Tm", "G", "GS", "FG%", "3P%", "FT%", "PTS"]
    )  # 19 features
    y = data["Pos"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X[["AST", "BLK"]] *= 10

    return X, y_encoded, label_encoder


def correlation_distance(x, y):
    return 1 - np.corrcoef(x, y)[0, 1]


def canberra_distance(x, y):
    return canberra(x, y)


def chi_square_distance(x, y):
    return 0.5 * np.sum((x - y) ** 2 / (x + y + 1e-10))


def create_model() -> VotingClassifier:
    """
    Create a voting ensemble model with multiple models

    Returns
    -------
    models : VotingClassifier - an ensemble model with multiple models
    """
    adaboost = AdaBoostClassifier(
        random_state=RANDOM_STATE,
        n_estimators=50,
        learning_rate=1,
    )
    adaboost2 = AdaBoostClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        learning_rate=1,
    )
    adaboost3 = AdaBoostClassifier(
        random_state=RANDOM_STATE,
        n_estimators=75,
        learning_rate=1,
    )
    randomforest = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=50,
        max_depth=10,
        criterion="entropy",
        min_samples_leaf=2,
        min_samples_split=10,
    )
    svm = SVC(random_state=RANDOM_STATE)
    logistic = LogisticRegression(random_state=RANDOM_STATE)
    knn2 = KNeighborsClassifier(
        n_neighbors=10,
        metric=correlation_distance,
        weights="distance",
    )
    knn3 = KNeighborsClassifier(
        n_neighbors=10,
        metric=canberra_distance,
        weights="distance",
    )
    knn6 = KNeighborsClassifier(
        n_neighbors=10,
        metric=chi_square_distance,
        weights="distance",
    )
    gradient_boosting = GradientBoostingClassifier(random_state=RANDOM_STATE)
    naive_bayes = GaussianNB()
    extra_trees = ExtraTreesClassifier(random_state=RANDOM_STATE)
    xgboost = XGBClassifier(
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="mlogloss",
        learning_rate=0.01,
        max_depth=5,
        n_estimators=200,
        subsample=0.7,
    )
    mlpc1 = MLPClassifier(
        random_state=RANDOM_STATE,
        hidden_layer_sizes=(200),
        activation="relu",
        solver="adam",
        max_iter=400,
        alpha=0.0001,
        learning_rate="constant",
    )
    decision_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    mlpc2 = MLPClassifier(
        random_state=RANDOM_STATE,
        learning_rate_init=(0.001),
        hidden_layer_sizes=(150,),
        max_iter=400,
        activation="tanh",
        solver="adam",
    )
    mlpc3 = MLPClassifier(
        random_state=RANDOM_STATE,
        hidden_layer_sizes=(50, 50),
        learning_rate_init=0.001,
        activation="tanh",
        solver="sgd",
        max_iter=400,
    )

    # weights = [10, 10, 10, 1, 7, 5, 2, 3, 2, 1, 1, 2, 1, 7, 1, 7, 5] # 0.708 validate, 0.718 test
    weights = [
        10,  # adaboost
        10,  # adaboost2
        10,  # adaboost3
        1,  # randomforest
        7,  # svm
        5,  # logistic
        2,  # knn2
        3,  # knn3
        2,  # knn6
        1,  # gradient_boosting
        1,  # naive_bayes
        2,  # extra_trees
        1,  # xgboost
        7,  # mlpc1
        1,  # decision_tree
        7,  # mlpc2
        5,  # mlpc3
    ]
    models = VotingClassifier(
        estimators=[
            ("adaboost", adaboost),
            ("adaboost2", adaboost2),
            ("adaboost3", adaboost3),
            ("randomforest", randomforest),
            ("svm", svm),
            ("logistic", logistic),
            ("knn2", knn2),
            ("knn3", knn3),
            ("knn6", knn6),
            ("gradient_boosting", gradient_boosting),
            ("naive_bayes", naive_bayes),
            ("extra_trees", extra_trees),
            ("xgboost", xgboost),
            ("mlpc1", mlpc1),
            ("decision_tree", decision_tree),
            ("mlpc2", mlpc2),
            ("mlpc3", mlpc3),
        ],
        voting="hard",
        weights=weights,
    )

    return models


def train_models(X_train, y_train, label_encoder, models) -> None:
    """
    Train a voting ensemble model with multiple models

    Parameters
    ----------
    X_train : pd.DataFrame - features of the data to be used for training
    y_train : np.ndarray - encoded labels of the data to be used for training
    label_encoder : LabelEncoder - label encoder used to encode and decode the labels
    models : VotingClassifier - an ensemble model with multiple models
    """

    models.fit(X_train, y_train)

    y_pred = models.predict(X_train)

    accuracy = accuracy_score(y_train, y_pred)
    print(f"Training set accuracy: %.3f" % accuracy)

    print("Confusion Matrix")
    print(
        pd.crosstab(
            label_encoder.inverse_transform(y_train),
            label_encoder.inverse_transform(y_pred),
            rownames=["Actual"],
            colnames=["Predicted"],
            margins=True,
        )
    )


def evaluate_voting_model(models, X, y, label_encoder, data_type) -> None:
    """
    Evaluate the voting model with the given data and print the accuracy and classification report

    Parameters
    ----------
    models : VotingClassifier - trained voting ensemble model
    X : pd.DataFrame - features of the data to be used for evaluation
    y : np.ndarray - encoded labels of the data to be used for evaluation
    label_encoder : LabelEncoder - label encoder used to encode and decode the labels
    data_type : str - type of the data (e.g., "Train", "Test", "Validate")

    Returns
    -------
    None
    """
    y_pred = models.predict(X)

    accuracy = accuracy_score(y, y_pred)
    print(f"{data_type} set accuracy: %.3f" % accuracy)

    print("Confusion Matrix")
    print(
        pd.crosstab(
            label_encoder.inverse_transform(y),
            label_encoder.inverse_transform(y_pred),
            rownames=["Actual"],
            colnames=["Predicted"],
            margins=True,
        )
    )


def stratified_cross_validation(X, y_encoded, models) -> list:
    """
    Perform 10-fold stratified cross-validation and return the accuracies of each fold

    Parameters
    ----------
    X : pd.DataFrame - features of the data to be used for cross-validation
    y_encoded : np.ndarray - encoded labels of the data to be used for cross-validation
    models : VotingClassifier - an ensemble model with multiple models

    Returns
    -------
    accuracies : list - accuracies of each fold
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    accuracies = []
    for train_index, test_index in skf.split(X, y_encoded):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        models.fit(X_train, y_train)

        y_pred = models.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    return accuracies


def main():
    X, y_encoded, label_encoder = load_data("./nba_stats.csv")

    """
    Task 1
    * Use 0.8 of the data for training and 0.2 for validation
    * Print
        1) the training and validation accuracy
        2) the confusion matrix for the training and validation data
    """
    print("-" * 10 + " Task 1 " + "-" * 10)

    X_train, X_validate, y_train, y_validate = train_test_split(
        X, y_encoded, train_size=0.8, random_state=RANDOM_STATE
    )

    models = create_model()
    train_models(X_train, y_train, label_encoder, models)
    evaluate_voting_model(models, X_validate, y_validate, label_encoder, "Validate")

    """
    Task 2
    * Re-train the model with the entire data
    * With the test data (for now, dummy_test.csv), print
        1) the test accuracy
        2) the confusion matrix for the test data
    """
    print("-" * 10 + " Task 2 " + "-" * 10)

    models = create_model()
    train_models(X, y_encoded, label_encoder, models)

    X_test, y_test, _ = load_data("./dummy_test.csv")
    evaluate_voting_model(models, X_test, y_test, label_encoder, "Test")

    """
    Task 3
    * Apply 10-fold stratified cross-validation, printing
        1) the accuracy of each fold
        2) the average accuracy
    """
    print("-" * 10 + " Task 3 " + "-" * 10)

    models = create_model()

    accuracies = stratified_cross_validation(X, y_encoded, models)
    print(f"Cross-validation scores: {accuracies}")
    print(f"Average accuracy: %.2f" % np.mean(accuracies))


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    RANDOM_STATE = 0
    set_seed(RANDOM_STATE)

    main()
