import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import plots
from dataset import dataset_windowed


def MyModel(input_shape, output_dim):

    X_input = tf.keras.Input(input_shape)

    # CONV -> MaxPool -> Batch Normalization
    X = tf.keras.layers.Conv1D(
        filters=16, kernel_size=5, activation=tf.nn.elu, name="conv0", padding="same"
    )(X_input)
    X = tf.keras.layers.MaxPool1D(pool_size=5, name="maxpool0")(X)
    X = tf.keras.layers.BatchNormalization(momentum=0.999, name="bn0")(X)

    X = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=20,
        activation=tf.nn.elu,
        name="conv1",
        padding="same",
        strides=2,
    )(X)
    X = tf.keras.layers.MaxPool1D(pool_size=5, name="maxpool1")(X)
    X = tf.keras.layers.BatchNormalization(axis=2, name="bn1")(X)

    # GRU
    X = tf.keras.layers.GRU(64, name="gru")(X)

    # FLATTEN X + FULLYCONNECTED
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.Dense(
        100,
        activation=tf.nn.elu,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        name="fc1",
    )(X)
    X = tf.keras.layers.BatchNormalization(momentum=0.999)(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(
        output_dim,
        activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        name="output",
    )(X)

    # Create the keras model
    model = tf.keras.Model(inputs=X_input, outputs=X, name="RNNModel")

    return model


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")

    # ######## decide if RETRAIN or not (at the first run the train is automatic)
    retrain = True
    # ######## PARAMETERS for the dataset load process
    k = 150
    stride = 15
    # ######## WHERE TO SAVE THE MODEL
    MODEL_DIRECTORY = "models/rnn"

    # load and split the dataset
    ds = dataset_windowed(K=k, stride=stride)

    X_train, X_test, y_train, y_test = train_test_split(
        ds.drop("app", axis=1),
        ds["app"].astype("category"),
        test_size=0.3,
        random_state=1234,
    )
    print(f"X_train size: {X_train.shape}")
    print(f"X_test size: {X_test.shape}")

    # generate labels that Keras can understand
    y_train_dumm, y_test_dumm = pd.get_dummies(y_train), pd.get_dummies(y_test)

    if retrain or not os.path.isfile(MODEL_DIRECTORY + "/model.h5"):

        # generation of the model
        model = MyModel((k, 1), len(ds["app"].unique()))

        # #### LEARNING PARAMETERS #####
        epochs = 25
        batch_size = 64
        initial_lr = 2e-3
        lr_decay_rate = 0.9

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
            monitor="val_accuracy",
            min_delta=1e-3,
            patience=5,
            restore_best_weights=True,
        )

        # reshape of training data
        X_train_reshaped = pd.DataFrame(X_train["packets_length_total"].to_list())
        X_train_reshaped = X_train_reshaped.to_numpy().reshape((-1, k, 1))

        # training
        history = model.fit(
            x=X_train_reshaped,
            y=y_train_dumm,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.25,
            callbacks=[early_stopping],
        )

        plots.train_val_history(history.history)
        # save the model for future use
        model.save(MODEL_DIRECTORY + "/model.h5")
    else:
        # load the model
        model = tf.keras.models.load_model(MODEL_DIRECTORY + "/model.h5")
        print("Model loaded successfully.")

    # summary of the model
    model.summary()

    # reshape of the test data
    X_test_reshaped = pd.DataFrame(X_test["packets_length_total"].to_list())
    X_test_reshaped = X_test_reshaped.to_numpy().reshape((-1, k, 1))
    # evaluation on test set
    loss, acc = model.evaluate(X_test_reshaped, y_test_dumm, batch_size=100)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {acc}")

    # prediction for the generation of the confusion matrix
    y_pred_dumm = pd.DataFrame(
        model.predict(X_test_reshaped, batch_size=100), columns=y_test_dumm.columns,
    )
    y_pred = y_pred_dumm.idxmax(axis="columns").astype("category")

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

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

    # confusion matrix
    plots.confusion_matrix_tf(y_test, y_pred)
    plots.show()

    # print the architecture
    # from tensorflow.keras.utils import plot_model
    # plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
