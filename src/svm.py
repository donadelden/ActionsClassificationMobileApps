import pandas as pd
import numpy as np
from dataset import dataset_mean_variance
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import plot_confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt


def model_svm(cost=1.0, kernel="rbf", gamma="scale", verbose=False, random_state=None):
    model = svm.SVC(
        decision_function_shape="ovr",
        C=cost,
        kernel=kernel,
        gamma=gamma,
        verbose=verbose,
        random_state=random_state,
    )
    return model


def model_svm_bagging(base_model, n_estimators, sampling_coeff=2, random_state=None):
    model = BaggingClassifier(
        base_model,
        n_estimators,
        max_samples=sampling_coeff / n_estimators,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
    )
    return model


def gridsearch_model(classifier, parameters):
    model = GridSearchCV(classifier, parameters, n_jobs=-1, refit=True, verbose=1)
    return model


if __name__ == "__main__":

    ds = dataset_mean_variance(agg_by="sequence", filter="both", na="drop")
    # ds = ds.query('app == "facebook" | app == "twitter"')

    X_train, X_test, y_train, y_test = train_test_split(
        ds.drop("app", axis=1),
        ds["app"],
        test_size=0.3,
        # train_size=0.2,
        random_state=1234,
    )

    parameters = [
        {
            "kernel": ("rbf",),
            "C": (10, 100, 1e3, 1e4, 1e5),
            "gamma": ("scale", "auto", 1e-6, 1e-5, 1e-4),
        }
    ]

    # classifier = model_svm(verbose=True, random_state=2345)
    # classifier = model_svm_bagging(model_svm(cost=100, kernel='rbf', gamma=1e-3), 8, random_state=2345)
    classifier = gridsearch_model(model_svm(), parameters)

    classifier.fit(X_train, y_train)
    classifier.verbose = 0

    res = pd.DataFrame(classifier.cv_results_)
    res = res.set_index("rank_test_score").sort_index()
    print(
        res[
            [
                "params",
                "mean_test_score",
                "std_test_score",
                "mean_fit_time",
                "std_fit_time",
            ]
        ]
    )

    print(classifier.score(X_test, y_test))

    precision, recall, fscore, support = precision_recall_fscore_support(
        y_test, classifier.predict(X_test)
    )

    val = (
        y_test.value_counts()
        .rename("support")
        .to_frame()
        .reset_index()
        .merge(
            pd.DataFrame(
                {
                    "precision": precision,
                    "recall": recall,
                    "fscore": fscore,
                    "support": support,
                }
            ),
            on="support",
        )
        .set_index("index")
    )

    print(val)

    plt.figure()
    plot_confusion_matrix(classifier, X_test, y_test, ax=plt.gca())
    plt.show()
