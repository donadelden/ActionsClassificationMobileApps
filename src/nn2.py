import pandas
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np

# import matplotlib.pyplot as plt
# from sklearn.metrics import ConfusionMatrixDisplay
from dataset import dataset_windowed


class FFNN(tf.keras.Model):
    def __init__(self, output_dim):
        super(FFNN, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(100, activation=tf.nn.elu)
        self.hidden2 = tf.keras.layers.Dense(100, activation=tf.nn.elu)
        self.hidden3 = tf.keras.layers.Dense(50, activation=tf.nn.elu)
        self.dropout = tf.keras.layers.Dropout(0.7)
        self.batchnorm = tf.keras.layers.BatchNormalization(momentum=0.999)
        self.last = tf.keras.layers.Dense(output_dim, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.hidden1(inputs)
        if training:
            x = self.batchnorm(x)
        x = self.hidden2(x)
        if training:
            x = self.dropout(x)
        x = self.hidden3(x)
        return self.last(x)


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    # load and split the dataset
    ds = dataset_windowed(N=100000, K=300)
    ingress = (
        ds["packets_length_total"]
        .map(np.array)
        .map(lambda l: l[l > 0])
        .where(lambda s: s.map(lambda l: l.size) > 0)
        .dropna()
    )
    ds["packets_length_mean"] = ingress.map(np.mean)
    ds["packets_length_std"] = ingress.map(np.std)
    ds = ds.drop(columns=["packets_length_total"]).dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        ds.drop("app", axis=1),
        ds["app"],
        test_size=0.2,
        # train_size=0.2,
        random_state=1234,
    )
    print("X_train size: ", X_train.shape)
    print("X_test size: ", X_test.shape)

    # generate labels that Keras can understand
    y_train_dumm, y_test_dumm = pandas.get_dummies(y_train), pandas.get_dummies(y_test)

    # generation of the model
    model = FFNN(len(ds["app"].unique()))

    model.compile(
        loss="categorical_crossentropy", optimizer="Adamax", metrics=["accuracy"]
    )

    history = model.fit(x=X_train, y=y_train_dumm, epochs=1, batch_size=50)

    model.summary()
    print(history.history)

    # evaluation on test set
    loss, acc = model.evaluate(X_test, y_test_dumm, batch_size=50)
    print(f"loss: {loss}")
    print(f"acc: {loss}")

    # y_pred = model.predict(X_test, batch_size=100)

    # cm = tf.math.confusion_matrix(labels=y_test_dumm, predictions=y_pred).numpy()
    # disp = ConfusionMatrixDisplay(cm, y_test_dumm.columns)
    # disp.plot()
