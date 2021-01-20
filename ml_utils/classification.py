"""Utilities for evaluating classification models."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc, average_precision_score, confusion_matrix,
    precision_recall_curve, r2_score, roc_curve
)

def confusion_matrix_visual(y_true, y_pred, class_labels, normalize=False,
                            flip=False, ax=None, title=None, **kwargs):
    """
    Create a confusion matrix heatmap to evaluate classification.

    Parameters:
        - y_test: The true values for y
        - preds: The predicted values for y
        - class_labels: What to label the classes.
        - normalize: Whether to plot the values as percentages.
        - flip: Whether to flip the confusion matrix. This is helpful to get
                TP in the top left corner and TN in the bottom right when dealing
                with binary classification with labels True and False.
        - ax: The matplotlib `Axes` object to plot on.
        - title: The title for the confusion matrix
        - kwargs: Additional keyword arguments for `seaborn.heatmap()`

    Returns:
        A matplotlib `Axes` object.
    """
    mat = confusion_matrix(y_true, y_pred)
    if normalize:
        fmt, mat = '.2%', mat / mat.sum()
    else:
        fmt = 'd'

    if flip:
        class_labels = class_labels[::-1]
        mat = np.flip(mat)

    axes = sns.heatmap(
        mat.T, square=True, annot=True, fmt=fmt,
        cbar=True, cmap=plt.cm.Blues, ax=ax, **kwargs
    )
    axes.set(xlabel='Actual', ylabel='Model Prediction')
    tick_marks = np.arange(len(class_labels)) + 0.5
    axes.set_xticks(tick_marks)
    axes.set_xticklabels(class_labels)
    axes.set_yticks(tick_marks)
    axes.set_yticklabels(class_labels, rotation=0)
    axes.set_title(title or 'Confusion Matrix')
    return axes

def find_threshold_roc(y_test, y_preds, *, fpr_below, tpr_above):
    """
    Find the threshold to use with `predict_proba()` for classification
    based on the maximum acceptable FPR and the minimum acceptable TPR.

    Parameters:
        - y_test: The actual labels.
        - y_preds: The predicted labels.
        - fpr_below: The maximum acceptable FPR.
        - tpr_above: The minimum acceptable TPR.

    Returns:
        The thresholds that produce a classification meeting the criteria.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_preds, drop_intermediate=False)
    return thresholds[(fpr <= fpr_below) & (tpr >= tpr_above)]

def find_threshold_pr(y_test, y_preds, *, min_precision, min_recall):
    """
    Find the threshold to use with `predict_proba()` for classification
    based on the minimum acceptable precision and the minimum acceptable recall.

    Parameters:
        - y_test: The actual labels.
        - y_preds: The predicted labels.
        - min_precision: The minimum acceptable precision.
        - min_recall: The minimum acceptable recall.

    Returns:
        The thresholds that produce a classification meeting the criteria.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_preds)
    # precision and recall have one extra value at the end for plotting
    # but this needs to be removed in order to make a mask with the thresholds
    return thresholds[(precision[:-1] >= min_precision) & (recall[:-1] >= min_recall)]

def plot_roc(y_test, preds, ax=None):
    """
    Plot ROC curve to evaluate classification.

    Parameters:
        - y_test: The true values for y
        - preds: The predicted values for y as probabilities
        - ax: The `Axes` object to plot on

    Returns:
        A matplotlib `Axes` object.
    """
    if not ax:
        fig, ax = plt.subplots(1, 1)

    fpr, tpr, thresholds = roc_curve(y_test, preds)

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='baseline')
    ax.plot(fpr, tpr, color='red', lw=2, label='model')

    ax.legend(loc='lower right')
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')

    ax.annotate(f'AUC: {auc(fpr, tpr):.2}', xy=(0.5, 0), horizontalalignment='center')

    return ax

def plot_pr_curve(y_test, preds, positive_class=1, ax=None):
    """
    Plot precision-recall curve to evaluate classification.

    Parameters:
        - y_test: The true values for y
        - preds: The predicted values for y as probabilities
        - positive_class: The label for the positive class in the data
        - ax: The matplotlib `Axes` object to plot on

    Returns:
        A matplotlib `Axes` object.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, preds)

    if not ax:
        fig, ax = plt.subplots()

    ax.axhline(
        sum(y_test == positive_class) / len(y_test),
        color='navy', lw=2, linestyle='--', label='baseline'
    )
    ax.plot(recall, precision, color='red', lw=2, label='model')

    ax.legend()
    ax.set_title(
        'Precision-recall curve\n'
        f"""AP: {average_precision_score(
            y_test, preds, pos_label=positive_class
        ):.2} | """
        f'AUC: {auc(recall, precision):.2}'
    )
    ax.set(xlabel='Recall', ylabel='Precision')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    return ax

def plot_multiclass_roc(y_test, preds, ax=None):
    """
    Plot ROC curve to evaluate classification.

    Parameters:
        - y_test: The true values for y
        - preds: The predicted values for y as probabilities
        - ax: The `Axes` object to plot on

    Returns:
        A matplotlib `Axes` object.
    """
    if not ax:
        fig, ax = plt.subplots(1, 1)

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='baseline')

    class_labels = np.sort(y_test.unique())
    for i, class_label in enumerate(class_labels):
        actuals = np.where(y_test == class_label, 1, 0)
        predicted_probabilities = preds[:,i]

        fpr, tpr, thresholds = roc_curve(actuals, predicted_probabilities)
        auc_score = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, label=f"""class {class_label}; AUC: {auc_score:.2}""")

    ax.legend()
    ax.set_title('Multiclass ROC curve')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')

    return ax

def plot_multiclass_pr_curve(y_test, preds):
    """
    Plot precision-recall curve to evaluate classification.

    Parameters:
        - y_test: The true values for y
        - preds: The predicted values for y as probabilities

    Returns:
        A matplotlib `Axes` object.
    """
    class_labels = np.sort(y_test.unique())

    row_count = np.ceil(len(class_labels) / 3).astype(int)
    fig, axes = plt.subplots(row_count, 3, figsize=(15, row_count*5))
    axes = axes.flatten()

    if len(axes) > len(class_labels):
        for i in range(len(class_labels), len(axes)):
            fig.delaxes(axes[i])

    for i, class_label in enumerate(class_labels):
        axes[i].axhline(sum(y_test == class_label)/len(y_test), color='navy', lw=2, linestyle='--', label='baseline')
        actuals = np.where(y_test == class_label, 1, 0)
        predicted_probabilities = preds[:,i]
        precision, recall, thresholds = precision_recall_curve(actuals, predicted_probabilities)
        auc_score = auc(recall, precision)
        ap_score = average_precision_score(actuals, predicted_probabilities)
        axes[i].plot(recall, precision, lw=2, label=f"""AUC: {auc_score:.2}; AP : {ap_score:.2}""")

        axes[i].legend()
        axes[i].set_title(f'Precision-recall curve: class {class_label}')
        axes[i].set(xlabel='Recall', ylabel='Precision')

        axes[i].set_xlim(-0.05, 1.05)
        axes[i].set_ylim(-0.05, 1.05)

    return axes
