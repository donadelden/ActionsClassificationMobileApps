import os
from dataset import dataset_mean_variance
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

MODELS_FOLDER = "models/rf"


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


def generate_models_for_actions(ds, retrain=False, grid_search=False, verbose=1):
    if retrain or not os.listdir(MODELS_FOLDER):
        # detect list of apps
        apps = ds["app"].unique()
        for app in apps:
            # generate the new dataset
            ds_new = ds[ds["app"] == app]
            ds_new = ds_new.drop("app", axis=1)

            # slit data
            X_train, X_test, y_train, y_test = train_test_split(
                ds_new.drop("action", axis=1),
                ds_new["action"],
                test_size=0.2,
                # train_size=0.2,
                random_state=1234,
            )
            if grid_search:
                n_estimators = [
                    int(x) for x in np.linspace(start=100, stop=1000, num=3)
                ]
                max_features = ["sqrt"]
                # max_depth = [int(x) for x in np.linspace(5, 15, num=3)]
                # max_depth.append(None)
                max_depth = [15]
                bootstrap = [True, False]
                max_samples = [0.1, 0.5, 1]

                parameters = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "max_features": [None],
                    # "bootstrap": bootstrap,
                    # "max_samples": max_samples,
                }

                classifier = gridsearch_model(model_rf(), parameters)
            else:
                classifier = model_rf(
                    max_depth=10,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    min_samples_split=2,
                    n_estimators=500,
                )

            print(f"Start fitting on action of {app}...")
            classifier.fit(X_train, y_train)
            print("done!")
            print(f"Result: {classifier.score(X_test, y_test)}.")
            if grid_search and verbose == 1:
                print(classifier.best_params_)

            filename = MODELS_FOLDER + "/" + app + "_model.sav"
            pickle.dump(classifier, open(filename, "wb"))
    else:
        print("Skipping training.")
    return True


def evaluate_all(test, size=None):
    correct = 0
    if size is not None:
        test = test.sample(size)
    for app in test["app_pred"].unique():
        # load the correct model
        loaded_model = pickle.load(open(MODELS_FOLDER + "/" + app + "_model.sav", "rb"))
        # predict actions
        app_samples = test[test["app_pred"] == app]
        actions_pred = loaded_model.predict(
            app_samples.drop(["action", "app_pred"], axis=1)
        )
        # concatenate
        pred = pd.DataFrame(app_samples["action"])
        pred["action_pred"] = actions_pred
        # compare the two columns
        correct += pred.apply(lambda x: x[0] == x[1], axis=1).value_counts()[True]

    return correct / test.shape[0]


if __name__ == "__main__":

    ds = dataset_mean_variance(filter="both", na="drop", agg_by="action")
    ds = ds.dropna()

    if not generate_models_for_actions(ds, retrain=False, grid_search=True, verbose=1):
        print("Error generating models, exit.")
        exit(1)

    # if "action" in ds:
    #    ds = ds.drop("app", axis=1)

    apps = ds["app"].unique()
    it = iter(range(len(apps)))
    m = {app: next(it) for app in apps}

    X_train, X_test, y_train, y_test = train_test_split(
        ds.drop("app", axis=1),
        ds["app"],
        test_size=0.2,
        # train_size=0.2,
        random_state=1234,
    )

    print(f"X_train size: {X_train.size}")
    print(f"X_test size: {X_test.size}")

    # Grid Search Parameters
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=1000, stop=3000, num=5)]
    # Number of features to consider at every split
    max_features = ["sqrt"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 200, num=1)]
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
        "max_features": [None],
        # "bootstrap": bootstrap,
        # "max_samples": max_samples,
    }

    # classifier = gridsearch_model(model_rf(), parameters)

    # Best parameters with ingress/egress division with new dataset pre computations:
    # 0.9844444444444445
    # {'max_depth': 64,  'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
    # Best parameters aggregating by actions
    # 0.8855958126571953
    # {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1500}

    classifier = model_rf(
        max_depth=10,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=1000,
    )

    # Train the model only if necessary
    if not os.path.isfile(MODELS_FOLDER + "/model.sav"):
        print("Start fitting...")
        classifier.fit(X_train.drop("action", axis=1), y_train)
        print("done!")
        filename = MODELS_FOLDER + "/model.sav"
        pickle.dump(classifier, open(filename, "wb"))
    else:
        classifier = pickle.load(open(MODELS_FOLDER + "/model.sav", "rb"))

    print(f"Test score: {classifier.score(X_test.drop('action', axis=1), y_test)}")
    # print(classifier.best_params_) # only for grid search

    # print confusion matrix
    # plt.figure()
    # plot_confusion_matrix(classifier, X_test.drop('action', axis=1), y_test, labels=apps, ax=plt.gca())
    # plt.show()

    test_all = X_test
    test_all["app_pred"] = classifier.predict(X_test.drop("action", axis=1))

    print("Begin of overall evaluation...")
    score = evaluate_all(test_all, size=1000)
    print(f"Score: {score}")
