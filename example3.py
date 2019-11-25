# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 06:19:19 2019

@author: hongj
"""

from generator.sample_generator import generator_multivariate_normal
from robustEM import rEM
from visualization import visualization_2d as vs2
import matplotlib.pyplot as plt


means= [[-4, -4], [-4, -4], [2, 2], [-1, -6]]
covs= [[[1, .5], [.5, 1]], 
       [[6, -2], [-2, 6]], 
       [[2, -1], [-1, 2]], 
       [[.125, .0], [.0, .125]]]
mix_prob= [.3, .3, .3, .1]

ex1= generator_multivariate_normal(means= means, 
                                   covs= covs,
                                   mix_prob= mix_prob)

X= ex1.get_sample(1000)

#Real
means_real= ex1.means_
covs_real= ex1.covs_

fig, ax= plt.subplots(1, 1)
vs2.get_figure(ax, X, means_real, covs_real, 'Real Data and Real Gaussian Distribution', 'b')

#robustEM
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
    vs2.get_figure(ax, X, result.means_, result.covs_, 'Iteration: %s; C: %s'%(result.iteration_, result.c_), 'r')


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