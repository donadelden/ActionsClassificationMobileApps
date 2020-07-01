import pandas as pd
import numpy as np
from dataset import dataset_mean_variance, dataset_windowed
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_recall_fscore_support
import plots


def model_svm(cost=1.0, kernel="rbf", gamma="scale", verbose=False, random_state=None):
    model = SVC(
        decision_function_shape="ovr",
        C=cost,
        kernel=kernel,
        gamma=gamma,
        verbose=verbose,
        random_state=random_state,
        max_iter=10000,
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


def get_mean_variance(ds):
    filtered_ingress = (
        ds["packets_length_total"]
        .map(np.array)
        .map(lambda l: l[l > 0])
        .where(lambda s: s.map(lambda l: l.size) > 0)
        .dropna()
    )
    filtered_egress = (
        ds["packets_length_total"]
        .map(np.array)
        .map(lambda l: -l[l < 0])
        .where(lambda s: s.map(lambda l: l.size) > 0)
        .dropna()
    )
    mean_ingress = filtered_ingress.map(np.mean).rename("ingress_packets_length_mean")
    variance_ingress = filtered_ingress.map(np.std).rename("ingress_packets_length_std")
    mean_egress = filtered_egress.map(np.mean).rename("egress_packets_length_mean")
    variance_egress = filtered_egress.map(np.std).rename("egress_packets_length_std")

    return pd.concat(
        [ds["app"], mean_ingress, variance_ingress, mean_egress, variance_egress],
        axis=1,
    ).dropna()


if __name__ == "__main__":

    # ds = dataset_mean_variance(agg_by="sequence", filter="both", na="drop")
    ds = dataset_windowed(K=175, stride=25)
    ds = get_mean_variance(ds)
    # ds = ds.query('app == "facebook" | app == "twitter"')

    X_train, X_test, y_train, y_test = train_test_split(
        ds.drop("app", axis=1),
        ds["app"],
        test_size=0.3,
        # train_size=0.2,
        random_state=1234,
    )

    # parameters = [
    #     {
    #         "kernel": ("rbf",),
    #         "C": (10, 100, 1e3, 1e4, 1e5),
    #         "gamma": ("scale", "auto", 1e-6, 1e-5, 1e-4),
    #     }
    # ]

    # classifier = model_svm(verbose=True, random_state=2345)
    # classifier = gridsearch_model(model_svm(), parameters)
    classifier = model_svm_bagging(model_svm(cost=100), 32, sampling_coeff=1.25)
    scaled_model = make_pipeline(StandardScaler(), classifier)

    scaled_model.fit(X_train, y_train)
    classifier.verbose = 0

    # res = pd.DataFrame(classifier.cv_results_)
    # res = res.set_index("rank_test_score").sort_index()
    # print(
    #     res[
    #         [
    #             "params",
    #             "mean_test_score",
    #             "std_test_score",
    #             "mean_fit_time",
    #             "std_fit_time",
    #         ]
    #     ]
    # )

    print(f"Test score: {scaled_model.score(X_test, y_test)}")

    precision, recall, fscore, support = precision_recall_fscore_support(
        y_test, scaled_model.predict(X_test)
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

    for base_classifier in classifier.estimators_:
        print(f"Support vectors: {sum(base_classifier.n_support_)}")

    plots.confusion_matrix_sk(scaled_model, X_test, y_test)

    plots.show()
