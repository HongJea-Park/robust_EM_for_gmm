# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 00:12:10 2019

@author: hongj
"""

from generator.sample_generator import generator_multivariate_normal
from robustEM import rEM
from visualization import visualization_2d as vs2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

means= [[.5, .5], 
        [2.5, .5], 
        [4.5, .5], 
        [6.5, .5],
        [.5, 2.5], 
        [2.5, 2.5], 
        [4.5, 2.5], 
        [6.5, 2.5], 
        [.5, 4.5], 
        [2.5, 4.5], 
        [4.5, 4.5], 
        [6.5, 4.5], 
        [.5, 6.5], 
        [2.5, 6.5], 
        [4.5, 6.5], 
        [6.5, 6.5]]

covs= [[[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]],
       [[.1, .0], [.0, .1]]]


mix_prob= [1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, \
           1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16]

ex1= generator_multivariate_normal(means= means, 
                                   covs= covs,
                                   mix_prob= mix_prob)

X= ex1.get_sample(800)

means_real= ex1.means_
covs_real= ex1.covs_

#Standard EM
gmm= GaussianMixture(n_components= 16, covariance_type= 'full')
gmm.fit(X)
means_sklearn= gmm.means_
covs_sklearn= gmm.covariances_

#robustEM
rem= rEM.robustEM()
rem.fit(X)

results= rem.result_list_
record= rem.save_record()

#visualization
plt.figure(figsize= (12, 6))
plt.subplots_adjust(wspace= .2, hspace= .5)

ax1, ax2, ax3, ax4, ax5, ax6= plt.subplot(231), plt.subplot(232), plt.subplot(233), \
                              plt.subplot(234), plt.subplot(235), plt.subplot(236)
                              
vs2.scatter_sample(ax1, X, 'Real Data and Real Gaussian Distribution')
vs2.get_figure(ax2, X, means_sklearn, covs_sklearn, 'Standard EM with sklearn', 'b')

ax_list= [ax3, ax4, ax5, ax6]
idx_= [1, 5, 20, -1]

for ax, idx in zip(ax_list, idx_):
    result= results[idx]
    vs2.get_figure(ax, X, result.means_, result.covs_, 'Iteration: %s; C: %s'%(result.iteration_, result.c_), 'r')
