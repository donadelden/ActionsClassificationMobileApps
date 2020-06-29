import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay


def train_val_history(history):
    fig, ax_left = plt.subplots()
    ax_left.set_xlabel("epoch")

    ax_left.set_ylabel("loss", color="tab:red")
    p_tl = ax_left.plot(history["loss"], color="tab:red", label="Train loss")
    p_vl = ax_left.plot(
        history["val_loss"], "--", color="tab:red", label="Validation loss"
    )
    ax_left.tick_params(axis="y", labelcolor="tab:red")

    ax_right = ax_left.twinx()

    ax_right.set_ylabel("accuracy", color="tab:blue")
    p_ta = ax_right.plot(history["accuracy"], color="tab:blue", label="Train Accuracy")
    p_va = ax_right.plot(
        history["val_accuracy"], "--", color="tab:blue", label="Validation Accuracy"
    )
    ax_right.tick_params(axis="y", labelcolor="tab:blue")

    # legend
    p_lines = p_tl + p_vl + p_ta + p_va
    p_labels = [line.get_label() for line in p_lines]
    plt.legend(p_lines, p_labels, loc=0)


def confusion_matrix_tf(y_true, y_pred):
    cm = tf.math.confusion_matrix(
        labels=y_true.cat.codes, predictions=y_pred.cat.codes
    ).numpy()
    disp = ConfusionMatrixDisplay(cm, display_labels=y_true.cat.categories)
    disp.plot()


def show():
    plt.show()
