import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import plots
from dataset import dataset_windowed


class FFNN(tf.keras.Model):
    def __init__(self, output_dim):
        super(FFNN, self).__init__()
        self.weight_regularizer = tf.keras.regularizers.l2(1e-3)

        self.conv1 = tf.keras.layers.Conv1D(
            filters=64, kernel_size=3, activation=tf.nn.relu
        )
        self.batchnorm1 = tf.keras.layers.BatchNormalization(momentum=0.999)
        self.flatten1 = tf.keras.layers.Flatten()

        self.conv2 = tf.keras.layers.Conv1D(
            filters=64, kernel_size=3, activation=tf.nn.relu
        )
        self.batchnorm2 = tf.keras.layers.BatchNormalization(momentum=0.999)
        self.flatten2 = tf.keras.layers.Flatten()

        self.hidden2 = tf.keras.layers.Dense(
            15, activation=tf.nn.sigmoid, kernel_regularizer=self.weight_regularizer
        )
        self.dropout2 = tf.keras.layers.Dropout(0.1)

        self.last = tf.keras.layers.Dense(output_dim, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.batchnorm1(x, training)
        x = self.flatten1(x)
        x = self.conv2(inputs)
        x = self.batchnorm2(x, training)
        x = self.flatten2(x)
        x = self.hidden2(x)
        x = self.dropout2(x, training)
        return self.last(x)


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    # load and split the dataset
    ds = dataset_windowed(K=200, stride=10)

    X_train, X_test, y_train, y_test = train_test_split(
        ds.drop("app", axis=1),
        ds["app"].astype("category"),
        test_size=0.2,
        random_state=1234,
    )
    print(f"X_train size: {X_train.shape}")
    print(f"X_test size: {X_test.shape}")

    # generate labels that Keras can understand
    y_train_dumm, y_test_dumm = pd.get_dummies(y_train), pd.get_dummies(y_test)

    # generation of the model
    model = FFNN(len(ds["app"].unique()))

    ##### LEARNING PARAMETERS #####
    epochs = 15
    batch_size = 32
    initial_lr = 1e-3
    lr_decay_rate = 0.97

    # variable learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=X_train.shape[0] // batch_size,
        decay_rate=lr_decay_rate,
        staircase=True,
    )

    optimizer = tf.keras.optimizers.Adamax(lr_schedule)

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", min_delta=1e-4, patience=5, restore_best_weights=True,
    )

    X_train_reshaped = pd.DataFrame(X_train["packets_length_total"].to_list())
    X_train_reshaped = X_train_reshaped.to_numpy().reshape((-1, 200, 1))

    history = model.fit(
        x=X_train_reshaped,
        y=y_train_dumm,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.25,
        callbacks=[early_stopping],
    )

    model.summary()

    plots.train_val_history(history.history)

    # evaluation on test set
    X_test_reshaped = pd.DataFrame(X_test["packets_length_total"].to_list())
    X_test_reshaped = X_test_reshaped.to_numpy().reshape((-1, 200, 1))
    loss, acc = model.evaluate(X_test_reshaped, y_test_dumm, batch_size=100)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {acc}")

    y_pred_dumm = pd.DataFrame(
        model.predict(X_test_reshaped, batch_size=100), columns=y_test_dumm.columns,
    )
    y_pred = y_pred_dumm.idxmax(axis="columns").astype("category")

    plots.confusion_matrix_tf(y_test, y_pred)
    plots.show()
