# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:37:30 2019

@author: hongj
"""

import matplotlib as mpl
import numpy as np
import matplotlib.transforms as transforms


def plot_sample(ax, X):
        
#    for i, s in enumerate(sets):
        
    ax.scatter(X[:, 0], X[:, 1], marker= '.', c= 'g', s= 10)
                  
        
def make_ellipses(ax, mean, cov, center, n_std):
    
    pearson = cov[0, 1]/ np.sqrt(cov[0, 0]* cov[1, 1])
    ell_radius_x= np.sqrt(1+ pearson)
    ell_radius_y= np.sqrt(1- pearson)
    
    ellipse= mpl.patches.Ellipse((0, 0), ell_radius_x* 2, ell_radius_y* 2, facecolor= 'none',
                                 edgecolor= 'r', lw= 1)
    
    scale_x= np.sqrt(cov[0, 0])* n_std
    scale_y= np.sqrt(cov[1, 1])* n_std
    
    transf= transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean[0], mean[1])
    
    ellipse.set_transform(transf+ ax.transData)
    ax.add_patch(ellipse)
    ax.add_artist(ellipse)
    ax.set_aspect('equal', 'datalim')
    
    if center:
        ax.scatter(mean[0], mean[1], c= 'black', marker= '*')
    
    
def get_figure(ax, sets, means, covs, center= False, n_std= 3):
    
    plot_sample(ax, sets)
    
    for mean, cov in zip(means, covs):
        make_ellipses(ax, mean, cov, center, n_std)
        
    