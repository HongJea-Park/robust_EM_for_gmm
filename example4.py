# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 04:17:25 2019

@author: hongj
"""

from generator.sample_generator import generator_univariate_normal
from robustEM import rEM
from visualization import visualization_1d as vs1
import matplotlib.pyplot as plt

means= [-11, 0, 13]
covs= [2, 4, 3]
mix_prob= [1/3, 1/3, 1/3]

ex1= generator_univariate_normal(means= means, 
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
                              
vs1.histogram(ax1, X, 'Real Data and Real Gaussian Distribution')

ax_list= [ax2, ax3, ax4, ax5]
idx_= [1, 10, 20, -1]

for ax, idx in zip(ax_list, idx_):
    result= results[idx]
    vs1.get_figure(ax, X, means, covs, mix_prob, results[idx], 'Iteration: %s; C: %s'%(result.iteration_, result.c_))

vs1.objective_function_plot(ax6, results)