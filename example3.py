# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 06:19:19 2019

@author: hongj
"""

from generator.sample_generator import generator_multivariate_normal
from robustEM import rEM
from visualization import visualization_2d as vs2
import matplotlib.pyplot as plt


#Data Generate
means= [[-4, -4], [-4, -4], [2, 2], [-1, -6]]
covs= [[[1, .5], [.5, 1]], 
       [[6, -2], [-2, 6]], 
       [[2, -1], [-1, 2]], 
       [[.125, .0], [.0, .125]]]
mix_prob= [.3, .3, .3, .1]

ex3= generator_multivariate_normal(means= means, 
                                   covs= covs,
                                   mix_prob= mix_prob)

X= ex3.get_sample(1000)


#Real
means_real= ex3.means_
covs_real= ex3.covs_

#fig, ax= plt.subplots(1, 1)
#vs2.scatter_sample(ax, X, 'Real Data and Real Gaussian Distribution')
#vs2.get_figure(ax, means_real, covs_real, 'b')
#
#plt.savefig('../plot/example3_1.png', dpi= 300)


#robust EM
rem= rEM.robustEM()
rem.fit(X)

results= rem.result_list_
record= rem.save_record()


#visualization
plt.figure(figsize= (12, 6))
plt.subplots_adjust(wspace= .2, hspace= .5)

ax_list= [plt.subplot(2, 3, i) for i in range(1, 7)]                              
idx_= [0, 1, 10, 20, 30, -1]

for ax, idx in zip(ax_list, idx_):
    result= results[idx]
    vs2.scatter_sample(ax, X, 'Iteration: %s; C: %s'%(result.iteration_, result.c_))
    vs2.get_figure(ax, means_real, covs_real, 'b')
    vs2.get_figure(ax, result.means_, result.covs_, 'tab:red')
    ax.legend(handles= [plt.plot([], ls= '-', color= 'b')[0],
                        plt.plot([], ls= '-', color= 'tab:red')[0]], \
              labels= ['R', 'E'], loc= 'lower left', fontsize= 7)


plt.savefig('../plot/example3_3.png', dpi= 300)


#Experiments by generating 100 data sets with same parameter
c_list= []

for _ in range(100):
    
    ex1= generator_multivariate_normal(means= means, 
                                       covs= covs,
                                       mix_prob= mix_prob)

    X= ex1.get_sample(1000)

    rem= rEM.robustEM()
    rem.fit(X)
    
    c_list.append(rem.c_== 4)

#the number of correct robustEM: 62