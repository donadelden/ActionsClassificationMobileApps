from dataset import dataset_mean_variance
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def model_svm(cost=1.0, kernel="rbf", verbose=False, random_state=None):
    model = svm.SVC(
        decision_function_shape="ovr",
        C=cost,
        kernel=kernel,
        verbose=verbose,
        random_state=random_state,
    )
    return model


def gridsearch_model(classifier, parameters):
    model = GridSearchCV(classifier, parameters, n_jobs=-1, refit=True, verbose=1)
    return model


if __name__ == "__main__":

    ds = dataset_mean_variance()[["app", "packets_length_mean", "packets_length_std"]]
    # ds = ds.query('app == "facebook" | app == "twitter"')

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

    parameters = {
        "kernel": ("rbf",),
        "C": (0.01, 0.1, 1, 10),
        "gamma": ("scale", "auto", 1e-4, 1e-3, 1e-2),
    }

    # classifier = model_svm(verbose=True, random_state=2345)
    classifier = gridsearch_model(model_svm(), parameters)

    classifier.fit(X_train, y_train)

    print(classifier.score(X_test, y_test))

    plt.figure()
    plot_confusion_matrix(classifier, X_test, y_test, labels=apps, ax=plt.gca())
    plt.show()
