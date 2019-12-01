# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 04:17:25 2019

@author: hongj
"""

from generator.sample_generator import generator_univariate_normal
from robustEM import rEM
from visualization import visualization_1d as vs1
from visualization import plot
import matplotlib.pyplot as plt


#Data Generate
means= [-11, 0, 13]
stds= [2, 4, 3]
mix_prob= [1/3, 1/3, 1/3]

ex4= generator_univariate_normal(means= means, 
                                 stds= stds,
                                 mix_prob= mix_prob)

X= ex4.get_sample(1000)


#Real
means_real= ex4.means_
stds_real= ex4.stds_


#robustEM
rem= rEM.robustEM()
rem.fit(X)

results= rem.result_list_
record= rem.save_record(save_option= True, filename= 'example4')


#visualization
fig= plt.figure(figsize= (12, 9))
fig.suptitle('Example4 with robust EM algorithm', fontsize= 16)
plt.subplots_adjust(wspace= .3, hspace= .3)

ax1, ax2, ax3, ax4, ax5, ax6= plt.subplot(231), plt.subplot(232), plt.subplot(233), \
                              plt.subplot(234), plt.subplot(235), plt.subplot(236)
                              
vs1.histogram(ax1, X, 'Real Data and Real Gaussian Distribution')

ax_list= [ax2, ax3, ax4, ax5]
idx_= [1, 10, 20, -1]

for ax, idx in zip(ax_list, idx_):
    result= results[idx]
    vs1.get_figure(ax, X, means, stds, mix_prob, results[idx], \
                   'Iteration: %s; C: %s'%(result.iteration_, result.c_))
    ax.legend(handles= [plt.plot([], ls= ':', color= 'b')[0],
                        plt.plot([], ls= '-', color= 'tab:red')[0]], \
              labels= ['R', 'E'], loc= 'upper right', fontsize= 7)

plot.objective_function_plot(ax6, results, 'darkorange')

plt.savefig('../plot/example4.png', dpi= 300)
