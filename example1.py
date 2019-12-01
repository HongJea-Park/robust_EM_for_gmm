# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 05:05:17 2019

@author: hongj
"""

from generator.sample_generator import generator_multivariate_normal
from robustEM import rEM
from visualization import visualization_2d as vs2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np


#Data Generate
means= [[.0, .0], [20, .0]]
covs= [[[1, .0], [.0, 1]], 
       [[9, .0], [.0, 9]]]
mix_prob= [.5, .5]

ex1= generator_multivariate_normal(means= means, 
                                   covs= covs,
                                   mix_prob= mix_prob)

X= ex1.get_sample(800)


#Real
means_real= ex1.means_
covs_real= ex1.covs_


#Standard EM with initial values
init_idx= np.random.choice(np.arange(X.shape[0]), 2)
means_init= X[init_idx, :]
gmm= GaussianMixture(n_components= 2, means_init= means_init)
gmm.fit(X)
means_sklearn= gmm.means_
covs_sklearn= gmm.covariances_
iteration_sklearn= gmm.n_iter_

plt.figure(figsize= (9, 4))
plt.subplots_adjust(wspace= .2)
ax1= plt.subplot(1, 2, 1)
vs2.scatter_sample(ax1, X, 'Real Data and Real Gaussian Distribution')
vs2.get_figure(ax1, means_real, covs_real, 'b', 'b', ls= ':')
ax2= plt.subplot(1, 2, 2)
vs2.scatter_sample(ax2, X, 'Standard EM with random initial values')
vs2.get_figure(ax2, means_sklearn, covs_sklearn, 'tab:red', 'tab:red')

plt.savefig('../plot/example1_1.png', dpi= 300)


#robust EM
rem= rEM.robustEM()
rem.fit(X)

results= rem.result_list_
record= rem.save_record()


#visualization
fig= plt.figure(figsize= (12, 9))
fig.suptitle('Example1 with robust EM algorithm', fontsize= 16)
plt.subplots_adjust(wspace= .2, hspace= .3)

ax_list= [plt.subplot(2, 3, i) for i in range(1, 7)]                              
idx_= [0, 1, 10, 20, 30, -1]

for ax, idx in zip(ax_list, idx_):
    result= results[idx]
    vs2.scatter_sample(ax, X, 'Iteration: %s; C: %s'%(result.iteration_, result.c_))
    vs2.get_figure(ax, means_real, covs_real, 'b', 'b', ls= ':')
    vs2.get_figure(ax, result.means_, result.covs_, 'tab:red', 'tab:red')

    ax.legend(handles= [plt.plot([], ls= ':', color= 'b')[0],
                        plt.plot([], ls= '-', color= 'tab:red')[0]], \
              labels= ['R', 'E'], loc= 'lower left', fontsize= 7)
    
plt.savefig('../plot/example1_2.png', dpi= 300)
