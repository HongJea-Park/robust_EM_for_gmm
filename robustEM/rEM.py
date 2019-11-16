# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:46:39 2019

@author: hongj
"""

import numpy as np
from scipy.stats import multivariate_normal
from time import time
import pandas as pd


class Result(object):
    
    pass


class robustEM():
    
    
    def __init__(self, X):
        
        self.X= X
        self.dim= X.shape[1]
        self.n= self.X.shape[0]
        self.gamma= 1e-4
        self.eps= 1e-6
        self.beta_update= True
        self.t= 0
        
        self.beta= 1
        self.c= self.n        
        self.pi= np.ones(self.c)/ self.c
        self.mus= self.X.copy()
        self.cov_idx= int(np.ceil(np.sqrt(self.c)))
        
        self.result_list= []
        
    
    def initialize_covmat(self):
        
        D_mat= np.sqrt(np.sum((self.X[None, :] - self.X[:, None])**2, -1))
        
        covs= np.apply_along_axis(func1d= lambda x: self._initialize_covmat_1d(x),
                                  axis= 1,
                                  arr= D_mat)
        
        self.covs= covs
        
        D_mat_reshape= D_mat.reshape(-1, 1)
        d_min= D_mat_reshape[D_mat_reshape> 0].min()
        
        self.Q= d_min* np.identity(self.dim)
    
    
    def _initialize_covmat_1d(self, d_k):
        
        d_k= d_k.copy()
        d_k.sort()
        d_k= d_k[d_k!= 0]
        
        return (d_k[self.cov_idx]* np.identity(self.dim))
        
    
    def predict_proba(self, X):
        
        likelihood= np.zeros((self.n, self.c))
        
        for i in range(self.c):
            
            dist= multivariate_normal(mean= self.mus[i],
                                      cov= self.covs[i])
            likelihood[:, i]= dist.pdf(X)
        
        numerator= likelihood* self.pi
        denominator= numerator.sum(axis= 1)[:, np.newaxis]
        z= numerator/ denominator
        
        return z


    def update_mu(self):
        
        mu_list= []
        for i in range(self.c):
            mu_list.append((self.X* self.z[:, i].reshape(-1, 1)).sum(axis= 0)/ self.z[:, i].sum())
            
        return np.array(mu_list)
        

    def update_pi(self):
        
        self.pi_EM= self.z.sum(axis= 0)/ self.n
        self.entropy= (self.pi*np.log(self.pi)).sum()
        
        return self.pi_EM+ self.beta* self.pi*(np.log(self.pi)- self.entropy)
        
        
    def update_beta(self):
        
        if self.beta_update:
            self.beta= np.min([self._left_term_of_beta_(), self._right_term_of_beta_()])
    
        
    def _left_term_of_beta_(self):
        
        power= np.trunc(self.dim/2- 1)
        
        eta= np.min([1, 0.5**(power)])
        
        left_term= np.exp(-eta* self.n* np.abs(self.new_pi- self.pi)).sum()/ self.c
        
        return left_term
    
    
    def _right_term_of_beta_(self):
        
        pi_EM= np.max(self.pi_EM)
        pi_old= np.max(self.pi)
        
        right_term= (1- pi_EM)/ (-pi_old* self.entropy)
        
        return right_term
    
    
    def update_c(self):
        
        idx_bool= self.pi>= 1/ self.n
        new_c= idx_bool.sum()
        
        pi= self.pi[idx_bool]
        self.pi= pi/ pi.sum()
        
        z= self.z[:, idx_bool]
        self.z= z/ z.sum(axis= 1).reshape(-1, 1)
        
        self.mus= self.mus[idx_bool, :]
        
        return new_c
        
    
    def update_cov(self):
        
        cov_list= []
        
        for i in range(self.new_c):
            
            new_cov= np.cov((self.X- self.mus[i, :]).T, aweights= (self.z[:, i]/ self.z[:, i].sum()))
            new_cov= (1- self.gamma)* new_cov- self.gamma* self.Q
            cov_list.append(new_cov)
            
        self.covs= np.array(cov_list)
        

    def calculate_diff(self):
        
        return np.max(np.sqrt(np.sum((self.new_mus- self.mus)** 2, axis= 1)))
        

    def fit(self):
        
        self.initialize_covmat()
        self.z= self.predict_proba(self.X)

        self.before_time= time()        
        self.get_iter_info()
        
        self.t+= 1
        self.num_update_c= 0
        
        while True:
                        
            self.mus= self.update_mu()
            self.new_pi= self.update_pi()
            self.update_beta()
            self.pi= self.new_pi
            self.new_c= self.update_c()
            
            if self.new_c== self.c: self.num_update_c+= 1
            
            if self.t>= 60 and self.num_update_c== 60: 
                
                self.beta= 0
                self.beta_update= False
                
            self.c= self.new_c
            self.update_cov()
            self.z= self.predict_proba(self.X)
            self.new_mus= self.update_mu()
            
            if self.calculate_diff()< self.eps: 
                
                self.mus= self.new_mus
                break
            
            self.mus= self.new_mus
            
            self.get_iter_info()
            
            self.t+= 1
            
        self.get_iter_info()
            
            
    def get_iter_info(self):
        
        result= Result()
        result.mus= self.mus
        result.covs= self.covs
        result.iteration= self.t
        result.c= self.c
        result.time= time()- self.before_time
        self.before_time= time()
        
        self.result_list.append(result)
        
        
    def save_record(self, save_option= False, filename= None):
        
        t, c, time= [], [], []
        
        for result in self.result_list:
            t.append(result.iteration)
            c.append(result.c)
            time.append(result.time)
            
        df= {}
        df['iteration']= t
        df['c']= c
        df['time']= time
        df= pd.DataFrame(df, columns= ['iteration', 'c', 'time'])
        
        if save_option:
            df.to_csv('../result/%s.csv'%filename, index= False, sep= ',')
        
        else:
            return df
        
if __name__== '__main__':
        
    pass