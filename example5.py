# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 06:34:04 2019

@author: hongj
"""

from generator.sample_generator import generator_multivariate_normal
from robustEM import rEM
from visualization import visualization_2d as vs2
from visualization import visualization_1d as vs1
import matplotlib.pyplot as plt


means= [[.0, .0], 
        [.0, .0], 
        [-1.5, 1.5], 
        [1.5, 1.5], 
        [0, -2]]
covs= [[[.01, .0], [.0, 1.25]], 
       [[8, .0], [.0, 8]], 
       [[.2, .0], [.0, .015]], 
       [[.2, .0], [.0, .015]],
       [[1., .0], [.0, .2]]]
mix_prob= [.2, .2, .2, .2, .2]

ex1= generator_multivariate_normal(means= means, 
                                   covs= covs,
                                   mix_prob= mix_prob)

X= ex1.get_sample(1000)

#Real
means_real= ex1.means_
covs_real= ex1.covs_

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

ax_list= [ax2, ax3, ax4, ax5]
idx_= [1, 5, 20, -1]

for ax, idx in zip(ax_list, idx_):
    result= results[idx]
    vs2.get_figure(ax, X, result.means_, result.covs_, 'Iteration: %s; C: %s'%(result.iteration_, result.c_), 'r')
        
vs1.objective_function_plot(ax6, results)