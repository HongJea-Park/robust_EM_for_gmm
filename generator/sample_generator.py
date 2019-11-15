# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from visualization import visualization as vs
import matplotlib.pyplot as plt

class generater():
    
    def __init__(self, n_components, means, covs, mix_prob):
        
        self.n_components= n_components
        self.means= np.array(means)
        self.covs= np.array(covs)
        self.mix_prob= mix_prob
        
    
    def get_sample(self, size):
        
        sample_list= []
        
        for n in range(0, self.n_components):
            
            mean= self.means[n]
            cov= self.covs[n]
            num_of_sample= int(np.around(size* self.mix_prob[n]))
            
            sample_list.append(np.random.multivariate_normal(mean= mean, cov= cov, size= num_of_sample))
            
        return sample_list
    
    

        
if __name__== '__main__':
    
    means= [[0, 0], [20, 0]]
    covs= [[[1, 0], [0, 1]], [[9, 0], [0, 9]]]
    mix_prob= [.5, .5]
    
    ex1= generater(n_components= 2, 
                   means= means, 
                   covs= covs,
                   mix_prob= mix_prob)
    
    ex1_sets= ex1.get_sample(800)
    
    means= ex1.means
    covs= ex1.covs

    fig, ax= plt.subplots(1, 1)
    vs.get_figure(ax, ex1_sets, means, covs)
