# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:37:30 2019

@author: hongj
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_sample_2d(ax, sets):
        
    for i, s in enumerate(sets):
        
        ax.scatter(s[:, 0], s[:, 1], marker= '.', c= 'g', s= 10)
                    

def make_ellipses(ax, mean, cov):
    
    ax.scatter(mean[0], mean[1], marker= 'o', c= 'black')
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi
    v = 3. * np.sqrt(2* v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1],
                              180 + angle, facecolor= 'none', edgecolor= 'r', lw= 1.2)
    ax.add_artist(ell)
    ax.set_aspect('equal', 'datalim')
    
    
def get_figure(ax, sets, means, covs):
    
    plot_sample_2d(ax, sets)
    
    for mean, cov in zip(means, covs):
        make_ellipses(ax, mean, cov)