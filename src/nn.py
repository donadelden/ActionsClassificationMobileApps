import pandas
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import dataset_mean_variance

if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    # load and split the dataset
    ds = dataset_mean_variance(filter="both", na="drop")
    apps = ds["app"].unique()

    X_train, X_test, y_train, y_test = train_test_split(
        ds.drop("app", axis=1),
        ds["app"],
        test_size=0.3,
        # train_size=0.2,
        random_state=1234,
    )
    print("X_train size: ", X_train.size)
    print("X_test size: ", X_test.size)

    # generate labels that Keras can understand
    y_train_dumm = pandas.get_dummies(y_train)
    y_test_dumm = pandas.get_dummies(y_test)

    # generation of the model
    num_features = X_train.shape[1]  # automatic detection of number of features
    inputs = keras.Input(shape=num_features)
    # x = layers.Dense(12, activation="relu")(inputs)
    # x = layers.Dense(10, activation="relu")(x)
    # x = layers.Dense(7, activation="relu")(x)
    outputs = layers.Dense(
        apps.size, activation="softmax", kernel_initializer="uniform"
    )(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
    )

    history = model.fit(x=X_train, y=y_train_dumm, epochs=15, batch_size=32)

    model.summary()
    print(history.history)

    # evaluation on test set
    loss, acc = model.evaluate(X_test, y_test_dumm, batch_size=32)
    print("loss: %.2f" % loss)
    print("acc: %.2f" % acc)
