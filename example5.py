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


#Data Generate
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

ex5= generator_multivariate_normal(means= means, 
                                   covs= covs,
                                   mix_prob= mix_prob)

X= ex5.get_sample(1000)


#Real
means_real= ex5.means_
covs_real= ex5.covs_


#robust EM
rem= rEM.robustEM()
rem.fit(X)

results= rem.result_list_
record= rem.save_record(save_option= True, filename= 'example5')


#visualization
plt.figure(figsize= (12, 6))
plt.subplots_adjust(wspace= .3, hspace= .5)

ax1, ax2, ax3, ax4, ax5, ax6= plt.subplot(231), plt.subplot(232), plt.subplot(233), \
                              plt.subplot(234), plt.subplot(235), plt.subplot(236)
                              
vs2.scatter_sample(ax1, X, 'Real Data and Real Gaussian Distribution')

ax_list= [ax2, ax3, ax4, ax5]
idx_= [1, 5, 20, -1]

for ax, idx in zip(ax_list, idx_):
    result= results[idx]
    vs2.scatter_sample(ax, X, 'Iteration: %s; C: %s'%(result.iteration_, result.c_))
    vs2.get_figure(ax, means_real, covs_real, 'b')
    vs2.get_figure(ax, result.means_, result.covs_, 'tab:red')
    ax.legend(handles= [plt.plot([], ls= '-', color= 'b')[0],
                        plt.plot([], ls= '-', color= 'tab:red')[0]], \
              labels= ['R', 'E'], loc= 'lower left', fontsize= 7)
    
vs1.objective_function_plot(ax6, results, 'darkorange')

plt.savefig('../plot/example5.png', dpi= 300)
