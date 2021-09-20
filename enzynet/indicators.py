"""Indicators."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import numpy as np

from enzynet import constants


def unalikeability(L: np.ndarray) -> float:
    """Returns unalikeability of a set of labels."""  # p.4 of Kader2007.
    if len(L) <= 1:
        return 0
    else:
        classes = np.unique(L)
        somme = 0
        for i_class in classes:
            n_temp = np.sum(L == i_class)
            S_temp = np.sum(L != i_class)
            somme += n_temp * S_temp
        somme /= len(L)*(len(L)-1)
        return somme


class Indicators(object):
    """Indicators used to assess the model.

    Parameters
    ----------
    y_true : NumPy array of size (n_samples, 1)
        Array of true labels.

    y_pred : NumPy array of size (n_samples, 1)
        Array of predicted labels.
    """
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Initialization."""
        self.n_samples = y_true.shape[0]
        self.n_classes = constants.N_CLASSES
        self.y_true = y_true
        self.y_pred = y_pred

    def confusion_matrix(self) -> None:
        """Confusion matrix."""
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        for i, j in zip(self.y_true, self.y_pred):
            self.confusion_matrix[i-1,j-1] += 1
        print('Confusion matrix: \n {0}'.format(np.array(self.confusion_matrix, dtype=int)))

        self.confusion_matrix_scaled = \
            self.confusion_matrix/np.sum(self.confusion_matrix, axis=1)[:, None]
        print('Scaled confusion matrix: \n {0}'.format(np.around(self.confusion_matrix_scaled, 3)))

    def accuracy(self) -> None:
        """Accuracy."""
        self.accuracy = np.trace(self.confusion_matrix)/self.n_samples
        print('Accuracy: {0:.3f}'.format(self.accuracy))

    def precision_per_class(self) -> None:
        """Precision per class."""
        self.precision_per_class = \
            np.diag(self.confusion_matrix/np.sum(self.confusion_matrix, axis=0))
        print('Precision per class: {0}'.format(np.around(self.precision_per_class, 3)))

    def recall_per_class(self) -> None:
        """Recall per class."""
        self.recall_per_class = np.diag(self.confusion_matrix_scaled)
        print('Recall per class: {0}'.format(np.around(self.recall_per_class, 3)))

    def f1_per_class(self) -> None:
        """F1 per class."""
        self.f1_per_class = \
            2 * self.precision_per_class * self.recall_per_class/(self.precision_per_class + self.recall_per_class)
        print('F1 per class: {0}'.format(np.around(self.f1_per_class, 3)))

    def macro_precision(self) -> None:
        """Macro precision."""
        self.macro_precision = np.sum(self.precision_per_class)/self.n_classes
        print('Macro precision: {0:.3f}'.format(self.macro_precision))

    def macro_recall(self) -> None:
        """Macro recall."""
        self.macro_recall = np.sum(self.recall_per_class)/self.n_classes
        print('Macro recall: {0:.3f}'.format(self.macro_recall))

    def macro_f1(self) -> None:
        """Macro F1 score."""
        self.macro_f1 = 2 * self.macro_precision * self.macro_recall/(self.macro_precision + self.macro_recall)
        print('Macro F1: {0:.3f}'.format(self.macro_f1))
