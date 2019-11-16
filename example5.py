# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 06:34:04 2019

@author: hongj
"""

from generator.sample_generator import generator
from robustEM import rEM
from visualization import visualization_2d as vs2
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

ex1= generator(means= means, 
               covs= covs,
               mix_prob= mix_prob)

X= ex1.get_sample(1000)


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
        vs2.get_figure(ax, X, result.mus, result.covs)