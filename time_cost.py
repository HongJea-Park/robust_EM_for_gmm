# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 00:54:17 2019

@author: hongj
"""

import pandas as pd
import matplotlib.pyplot as plt
from visualization import plot

result_list= ['example4', 'example5', 'example7']

fig= plt.figure(figsize= (12, 5))
fig.suptitle('Computation time cost per each iteration in example 4, 5, 7', fontsize= 16)
ax_list= [plt.subplot(131), plt.subplot(132), plt.subplot(133)]
plt.subplots_adjust(wspace= .3)

for ax, result in zip(ax_list, result_list):
    df= pd.read_csv('../result/%s.csv'%result)
    plot.time_cost_plot(ax, df, 'Time cost in %s'%result)
    
plt.savefig('../plot/time_cost.png', dpi= 300)
