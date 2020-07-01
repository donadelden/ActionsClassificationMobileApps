import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


def confusion_matrix_sk(model, X, y_true):
    plt.figure(figsize=(5.3, 3.7))
    plot_confusion_matrix(model, X, y_true, ax=plt.gca())
    plt.gca().set_xticklabels([])


def show():
    plt.show()
