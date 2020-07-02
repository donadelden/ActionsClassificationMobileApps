import time
from dataset import dataset_windowed
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import plots
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
        n_jobs=-1,
    )
    return model


def gridsearch_model(classifier, parameters):
    model = GridSearchCV(classifier, parameters, n_jobs=-1, refit=True, verbose=1)
    return model


if __name__ == "__main__":

    # #### decide if Grid Search is needed (slow)
    grid = False
    # #### DATASET PARAMETERS
    k = 200
    stride = 10
    # generate the dataset
    ds = dataset_windowed(K=k, stride=stride)

    # preprocessing
    ingress = (
        ds["packets_length_total"]
        .map(np.array)
        .map(lambda l: l[l > 0])
        .where(lambda s: s.map(lambda l: l.size) > 0)
        .dropna()
    )
    ds["packets_length_mean_ingress"] = ingress.map(np.mean)
    ds["packets_length_std_ingress"] = ingress.map(np.std)

    egress = (
        ds["packets_length_total"]
        .map(np.array)
        .map(lambda l: -l[l < 0])
        .where(lambda s: s.map(lambda l: l.size) > 0)
        .dropna()
    )
    ds["packets_length_mean_egress"] = egress.map(np.mean)
    ds["packets_length_std_egress"] = egress.map(np.std)
    ds = ds.drop(columns=["packets_length_total"]).dropna()

    # splitting of the data
    X_train, X_test, y_train, y_test = train_test_split(
        ds.drop("app", axis=1),
        ds["app"],
        test_size=0.3,
        # train_size=0.2,
        random_state=1234,
    )
    print(f"X_train size: {X_train.shape}")
    print(f"X_test size: {X_test.shape}")

    if grid:
        # ##### PARAMETERS for Grid Search ###
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=10, stop=1500, num=4)]
        # Number of features to consider at every split
        max_features = ["auto", "sqrt", "log2"]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 100, num=3)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # If bootstrap is True, the number of samples to draw from X to train each base estimator.
        max_samples = [0.1, 0.5, 1]

        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "max_features": max_features,
            "bootstrap": bootstrap,
            # "max_samples": max_samples,
        }

        # grid search
        classifier = gridsearch_model(model_rf(), parameters)

    else:
        # use the best parameters
        classifier = model_rf(
            bootstrap=True,
            max_depth=50,
            max_features="sqrt",
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=100,
        )

    # training
    print("Begin training...")
    start = time.time()
    classifier.fit(X_train, y_train)
    stop = time.time()
    print(f"Done in {stop-start} seconds.")

    # print the score
    print(f"Score: {classifier.score(X_test, y_test)}")

    # if grid search, print also the best parameters
    if grid:
        print(classifier.best_params_)

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

    # plot confusion matrix
    plots.confusion_matrix_sk(classifier, X_test, y_test)
    plots.show()
