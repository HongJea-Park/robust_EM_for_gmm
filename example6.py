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
import numpy as np

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

ex6= generator_multivariate_normal(means= means, 
                                   covs= covs,
                                   mix_prob= mix_prob)

X= ex6.get_sample(800)

means_real= ex6.means_
covs_real= ex6.covs_

#Standard EM
init_idx= np.random.choice(np.arange(X.shape[0]), 16)
means_init= X[init_idx, :]
gmm= GaussianMixture(n_components= 16, means_init= means_init, max_iter= 1000, tol= 1e-10)
gmm.fit(X)
means_sklearn= gmm.means_
covs_sklearn= gmm.covariances_
iteration_sklearn= gmm.n_iter_

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
vs2.scatter_sample(ax2, X, 'Standard EM with sklearn; Iteration: %s'%iteration_sklearn)
vs2.get_figure(ax2, means_sklearn, covs_sklearn, 'b')

ax_list= [ax3, ax4, ax5, ax6]
idx_= [1, 5, 20, -1]

for ax, idx in zip(ax_list, idx_):
    result= results[idx]
    vs2.scatter_sample(ax, X, 'Iteration: %s; C: %s'%(result.iteration_, result.c_))
    vs2.get_figure(ax, result.means_, result.covs_, 'tab:red')

plt.savefig('../plot/example6.png', dpi= 300)
