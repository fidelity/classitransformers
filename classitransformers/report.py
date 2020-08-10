# Copyright 2020 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


def plot_confusion_matrix(cm, classes = [0, 1], cmap=plt.cm.Accent):

    """ function to plot confusion matrix """

    title = 'Confusion matrix'
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
    
def metrics(y, y_pred, average='macro'):
    
    """
    uses sklearn metrics to calculate accuracy, precision, recall, f1_score.

    Parameters
    ----------
    y : actual labels. array-like of shape (n_samples,)
    y_pred : predicted labels. array-like of shape (n_samples,)
    average: determines the type of averaging performed on the data. default: macro
    """

    cm = confusion_matrix(y, y_pred)
    plot_confusion_matrix(cm, np.arange(cm.shape[0]))
    print("\n")
    print("Accuracy: {0:.3f}".format(accuracy_score(y, y_pred)))
    print("Precision: {0:.3f}".format(precision_score(y, y_pred, average=average)))
    print("Recall: {0:.3f}".format(recall_score(y, y_pred, average=average)))
    print("F1-Score: {0:.3f}".format(f1_score(y, y_pred, average=average)))