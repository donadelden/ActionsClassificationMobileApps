from dataset import dataset_mean_variance
from dataset import dataset_mean_variance2
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def model_rf(
    n_estimators=100,
    max_depth=None,
    bootstrap=True,
    max_samples=None,
    max_features="auto",
    min_samples_leaf=1,
    min_samples_split=2,
):
    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        bootstrap=bootstrap,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_samples=max_samples,
    )
    return model


def gridsearch_model(classifier, parameters):
    model = GridSearchCV(classifier, parameters, n_jobs=-1, refit=True, verbose=1)
    return model


if __name__ == "__main__":

    # ds = dataset_mean_variance()[["app", "packets_length_mean", "packets_length_std"]]
    ds = dataset_mean_variance2()[
        [
            "app",
            "packets_length_mean_ingress",
            "packets_length_std_ingress",
            "packets_length_mean_egress",
            "packets_length_std_egress",
        ]
    ]
    # ds = ds.query('app == "facebook" | app == "twitter"')

    ds = ds.dropna()

    apps = ds["app"].unique()
    it = iter(range(len(apps)))
    m = {app: next(it) for app in apps}

    X_train, X_test, y_train, y_test = train_test_split(
        ds.drop("app", axis=1),
        ds["app"],
        test_size=0.2,
        train_size=0.2,
        random_state=1234,
    )

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=500, stop=1000, num=5)]
    # Number of features to consider at every split
    max_features = ["auto", "sqrt"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(60, 100, num=5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # If bootstrap is True, the number of samples to draw from X to train each base estimator.
    max_samples = [0.1, 0.5, 1, "None"]  # usefull only if bootstrap = False

    parameters = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        # "bootstrap": bootstrap,
        # "max_samples": max_samples,
    }

    # classifier = gridsearch_model(model_rf(), parameters)

    # Best parameters without ingress/egress division:
    # 0.7866372518121652
    # {'max_depth': 110, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    # Best parameters with ingress/egress division:
    # 0.8762830534973829
    # {'max_depth': 100, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1000}
    classifier = model_rf(
        max_depth=100,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=1000,
    )

    classifier.fit(X_train, y_train)

    print(classifier.score(X_test, y_test))
    # print(classifier.best_params_)

    plt.figure()
    plot_confusion_matrix(classifier, X_test, y_test, labels=apps, ax=plt.gca())
    plt.show()
