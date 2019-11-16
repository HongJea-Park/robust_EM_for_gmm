# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np


class generator():
    
    
    def __init__(self, means, covs, mix_prob):
        
        self.n_components= len(means)
        self.dim= len(means[0])
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
            
        X= np.zeros((0, self.dim))
        
        for sample in sample_list:    
            
            X= np.vstack([X, sample])
            
        return X
    

if __name__== '__main__':
    
    pass