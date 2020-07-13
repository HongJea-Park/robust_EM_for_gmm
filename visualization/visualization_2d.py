# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:37:30 2019

@author: hongj
"""

import matplotlib as mpl
import numpy as np
import matplotlib.transforms as transforms


def scatter_sample(ax, X, title):

    '''
        Scatter data sets in 2 dimention vector space.

    Args:
        ax: matplotlib.axes.Axes
        X: numpy array
        title: string
    '''

    ax.set_title(title, fontsize=10)
    ax.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
    ax.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
    ax.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)


def make_ellipses(ax, mean, cov, edgecolor, m_color, ls, n_std):

    '''
        Plot ellipses with mean vector and covariance matrix information.

    Args:
        ax: matplotlib.axes.Axes
        mean: numpy array
        cov: numpy array
        n_std: integer or float
        edgecolor: string
    '''

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = mpl.patches.Ellipse((0, 0),
                                  ell_radius_x * 2,
                                  ell_radius_y * 2,
                                  facecolor='none',
                                  edgecolor=edgecolor, lw=1, ls=ls)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y)
    transf = transf.translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    ax.add_artist(ellipse)
    ax.set_aspect('equal', 'datalim')

    ax.scatter(mean[0], mean[1], c=m_color, marker='*')


def get_figure(ax, means, covs, edgecolor, m_color, ls='-', n_std=3):

    '''
        Function for convenient visualization.

    Args:
        ax: matplotlib.axes.Axes
        X: numpy array
        mean: numpy array
        cov: numpy array
        edgecolor: string
        n_std: integer or float
    '''

    for mean, cov in zip(means, covs):
        make_ellipses(ax, mean, cov, edgecolor, m_color, ls, n_std)
