"""Plots for visualizing PCA."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def pca_scatter(X, labels, cbar_label, color_map='brg'):
    """
    Create a 2D scatter plot from 2 PCA components of X

    Parameters:
        - X: The X data for PCA
        - labels: The y values
        - cbar_label: The label for the colorbar
        - color_map: Name of the color_map to use. Default is 'brg'

    Returns:
        Matplotlib Axes object
    """
    pca = Pipeline([('scale', MinMaxScaler()), ('pca', PCA(2, random_state=0))]).fit(X)
    data = pca.transform(X)
    ax = plt.scatter(
        data[:, 0], data[:, 1],
        c=labels, edgecolor='none', alpha=0.5,
        cmap=plt.cm.get_cmap(color_map, 2)
    )
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    cbar.set_ticks([0, 1])
    plt.legend(
        ['explained variance\n'
        'comp. 1: {:.3}\ncomp. 2: {:.3}'.format(
            *pca.named_steps['pca'].explained_variance_ratio_
        )]
    )
    return ax

def pca_scatter_3d(X, labels, cbar_label, color_map='brg', elev=10, azim=15):
    """
    Create a 3D scatter plot from 3 PCA components of X

    Parameters:
        - X: The X data for PCA
        - labels: The y values
        - cbar_label: The label for the colorbar
        - color_map: Name of the color_map to use. Default is 'brg'
        - elev: The degrees of elevation to view the plot from. Default is 10.
        - azim: The azimuth angle on the xy plane (rotation around the z-axis). Default is 15.

    Returns:
        Matplotlib Axes object
    """
    pca = Pipeline([('scale', MinMaxScaler()), ('pca', PCA(3, random_state=0))]).fit(X)
    data = pca.transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter3D(
        data[:, 0], data[:, 1], data[:, 2], alpha=0.5,
        c=labels, cmap=plt.cm.get_cmap(color_map, 2)
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.set_zlabel('component 3')
    cbar = fig.colorbar(p)
    cbar.set_ticks([0, 1])
    cbar.set_label(cbar_label)
    plt.legend(
        ['explained variance\n'
        'comp. 1: {:.3}\ncomp. 2: {:.3}\ncomp. 3: {:.3}'.format(
            *pca.named_steps['pca'].explained_variance_ratio_
        )]
    )
    return ax

def pca_explained_variance_plot(pca_model, ax=None):
    """
    Plot the cumulative explained variance of PCA components.

    Parameters:
        - pca_model: The PCA model that has been fit already
        - ax: Matplotlib Axes to plot on.

    Returns:
        A matplotlib Axes objects
    """
    if not ax:
        fig, ax = plt.subplots()
    ax.plot(
        np.append(
            0, pca_model.explained_variance_ratio_.cumsum()
        ), 'o-'
    )
    ax.set_title('Total Explained Variance Ratio for PCA Components')
    ax.set_xlabel('PCA components used')
    ax.set_ylabel('cumulative explained variance ratio')

    return ax

def pca_scree_plot(pca_model, ax=None):
    """
    Plot the explained variance of each consecutive PCA component.

    Parameters:
        - pca_model: The PCA model that has been fit already
        - ax: Matplotlib Axes to plot on.

    Returns:
        A matplotlib Axes objects
    """
    if not ax:
        fig, ax = plt.subplots()

    ax.plot(np.arange(1, 9), pca_model.explained_variance_, 'o-')
    ax.set_title('Scree Plot for PCA Components')
    ax.set_xlabel('PCA components used')
    ax.set_ylabel('explained variance')

    return ax
