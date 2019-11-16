# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 06:10:47 2019

@author: hongj
"""

from generator.sample_generator import generator
from robustEM import rEM
from visualization import visualization_2d as vs2
import matplotlib.pyplot as plt


means= [[.0, 3], [.0, 5], [.0, 7]]
covs= [[[1.2, .0], [.0, .01]], 
       [[1.2, .0], [.0, .01]], 
       [[1.2, .0], [.0, .01]]]
mix_prob= [1/3, 1/3, 1/3]

ex1= generator(means= means, 
               covs= covs,
               mix_prob= mix_prob)

X= ex1.get_sample(300)


#Real
mus_real= ex1.means
covs_real= ex1.covs

fig, ax= plt.subplots(1, 1)
vs2.get_figure(ax, X, mus_real, covs_real)


#robustEM
rem= rEM.robustEM(X)
rem.fit()

results= rem.result_list
rem.save_record()

iter_list= [0, 1, 10, 20, 30, 51]

for i, result in enumerate(results):
    
    if i in iter_list:
        fig, ax= plt.subplots(1, 1)
        vs2.get_figure(ax, X, result.mus, result.covs, center= True)