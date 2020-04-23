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
    
    ''' 
        Class for recording iteration results.
    '''
    
    pass



class robustEM():
    
    '''
        Class for robust EM clustering algorithm.
        
    Args:
        gamma: float. Non-negative regularization added to the diagonal of 
            covariance. 
            This variable is equivalent to 'reg_covar' in 
            sklearn.mixture.GaussianMixture.
        eps: float. The convergence threshold.
             This variable is equivalent to 'tol' in 
             sklearn.mixture.GaussianMixture.
    '''
    
    def __init__(self, gamma= 1e-4, eps= 1e-3):
        
        self.gamma_ = gamma
        self.eps_ = eps
        self.smoothing_parameter_ = 1e-256
        
        self.result_list_ = []
        
    
    def fit(self, X):
        
        ''' 
            Function for training model with data X by using 
            robust EM algorithm.
            Refer 'Robust EM clustering algorithm' section in the paper 
            page 4 for more details.
        
        Args:
            X: numpy array
            
        '''
        
        if X.ndim == 1: self.X_ = X.reshape(-1, 1)
        else: self.X_ = X
        
        self.dim_ = self.X_.shape[1]
        self.n_ = self.X_.shape[0]
        self.c_ = self.n_
        self.pi_ = np.ones(self.c_) / self.c_
        self.means_ = self.X_.copy()
        self.cov_idx_ = int(np.ceil(np.sqrt(self.c_)))
        self.beta_ = 1
        self.beta_update_ = True
        self.t_ = 0
        self.entropy_ = (self.pi_*np.log(self.pi_)).sum()

        self.initialize_covmat()
        self.z_ = self.predict_proba(self.X_)

        self.before_time_ = time()        
        self.get_iter_info()
        
        self.t_ += 1
        self.num_update_c_ = 0
        
        while True:
                        
            self.means_ = self.update_means()
            self.new_pi_ = self.update_pi()
            self.update_beta()
            self.pi_ = self.new_pi_
            self.new_c_ = self.update_c()
            
            if self.new_c_ == self.c_: self.num_update_c_ += 1
            
            if self.t_ >= 60 and self.num_update_c_ == 60: 
                
                self.beta_ = 0
                self.beta_update_ = False
                
            self.c_ = self.new_c_
            self.update_cov()
            self.z_ = self.predict_proba(self.X_)
            self.new_means_ = self.update_means()
            
            if self.check_convergence() < self.eps_: 
                
                self.means_ = self.new_means_
                break
            
            self.means_ = self.new_means_
            
            self.get_iter_info()
            
            self.t_ += 1
            
        self.get_iter_info()

    
    def initialize_covmat(self):
        
        ''' 
            Covariance matrix initialize function.
        '''
        
        D_mat = np.sqrt(
            np.sum((self.X_[None, :] - self.X_[:, None]) ** 2, -1))
        
        self.covs_ = np.apply_along_axis(
            func1d= lambda x: self._initialize_covmat_1d(x),
            axis= 1,
            arr= D_mat)
        
        D_mat_reshape = D_mat.reshape(-1, 1)
        d_min = D_mat_reshape[D_mat_reshape > 0].min()
        
        self.Q_ = d_min* np.identity(self.dim_)
    
    
    def _initialize_covmat_1d(self, d_k):
        
        '''
            Function for self.initialize_covmat() that uses 
            np.apply_along_axis().
            This function is refered term 27 in the paper.
            
        Args:
            d_k: numpy 1d array
        '''
        d_k = d_k.copy()
        d_k.sort()
        d_k = d_k[d_k != 0]
        
        return ((d_k[self.cov_idx_] ** 2) * np.identity(self.dim_))
        
    
    def predict_proba(self, X):
        
        '''
            Function to calculate posterior probability of each component 
            given the data.
            
        Args:
            X: numpy array
        '''
        
        likelihood = np.zeros((self.n_, self.c_))
        
        for i in range(self.c_):
            
            self.covs_[i] = self.check_positive_semidefinite(self.covs_[i])
            
            dist = multivariate_normal(mean= self.means_[i],
                                       cov= self.covs_[i])
            likelihood[:, i] = dist.pdf(X)
        
        numerator = likelihood * self.pi_
        denominator = numerator.sum(axis= 1)[:, np.newaxis]
        z = numerator / denominator
        
        return z
    
    
    def predict(self, X):
        
        '''
            Function to predict the labels for the data samples in X.
            
        Args:
            X: numpy array
        '''
        
        argmax = self.predict_proba(X)
        
        return argmax.argmax(axis= 1)


    def update_means(self):
        
        '''
            Mean vectors update step.
            This function is refered term 25 in the paper.
        '''
        
        means_list = []
        for i in range(self.c_):
            means_list.append(
                (self.X_* self.z_[:, i].reshape(-1, 1)).sum(axis= 0) / \
                    self.z_[:, i].sum())
            
        return np.array(means_list)
        

    def update_pi(self):
        
        '''
            Mixing proportions update step.
            This function is refered term 13 in the paper.
        '''
        
        self.pi_EM_ = self.z_.sum(axis= 0) / self.n_
        self.entropy_ = (self.pi_*np.log(self.pi_)).sum()
        
        return self.pi_EM_ + self.beta_ * self.pi_ * \
            (np.log(self.pi_) - self.entropy_)
        
        
    def update_beta(self):
        
        '''
            Beta update step.
            This function is refered term 24 in the paper.
        '''
        
        if self.beta_update_:
            self.beta_ = np.min([
                self._left_term_of_beta_(), 
                self._right_term_of_beta_()])
    
        
    def _left_term_of_beta_(self):
        
        '''
            Left term of beta update step.
            This function is refered term 22 in the paper.
        '''
        
        power = np.trunc(self.dim_ / 2 - 1)
        
        eta = np.min([1, 0.5 ** (power)])
        
        left_term = np.exp(
            -eta * self.n_ * np.abs(self.new_pi_ - self.pi_)).sum() / self.c_
        
        return left_term
    
    
    def _right_term_of_beta_(self):
        
        '''
            Right term of beta update step.
            This function is refered term 23 in the paper.
        '''
        
        pi_EM = np.max(self.pi_EM_)
        pi_old = np.max(self.pi_)
        
        right_term = (1 - pi_EM) / (-pi_old * self.entropy_)
        
        return right_term
    
    
    def update_c(self):
        
        '''
            Update the number of components.
            This function is refered term 14, 15 and 16 in the paper.
        '''
        
        idx_bool = self.pi_ >= 1 / self.n_
        new_c = idx_bool.sum()
        
        pi = self.pi_[idx_bool]
        self.pi_ = pi / pi.sum()
        
        z = self.z_[:, idx_bool]
        self.z_ = z / z.sum(axis= 1).reshape(-1, 1)
        
        self.means_ = self.means_[idx_bool, :]
        
        return new_c
        
    
    def update_cov(self):
        
        '''
            Covariance matrix update step.
            This function is refered term 26 and 28 in the paper.
        '''
        
        cov_list = []
        
        for i in range(self.new_c_):
            
            new_cov = np.cov((self.X_- self.means_[i, :]).T, 
                             aweights= (self.z_[:, i]/ self.z_[:, i].sum()))
            new_cov = (1- self.gamma_)* new_cov- self.gamma_* self.Q_
            cov_list.append(new_cov)
            
        self.covs_ = np.array(cov_list)
        

    def check_convergence(self):
        
        '''
            Function for checking whether algorithm converge or not.
        '''
        
        return np.max(
            np.sqrt(np.sum((self.new_means_- self.means_)** 2, axis= 1)))
    
    
    def check_positive_semidefinite(self, cov):
        
        '''
            Function for preventing error that covariance matrix is not 
            positive semi definite.
        '''
        
        min_eig = np.min(np.linalg.eigvals(cov))
        
        if min_eig < 0:
            cov -= 10 * min_eig * np.eye(*cov.shape)
            
        return cov
        
            
    def get_iter_info(self):
        
        '''
            Record function that saves useful information in each step for 
            visualization and objective function.
        '''
        
        result = Result()
        result.means_ = self.means_
        result.covs_ = self.covs_
        result.iteration_ = self.t_
        result.c_ = self.c_
        result.time_ = time()- self.before_time_
        result.mix_prob_ = self.pi_
        result.beta_ = self.beta_
        result.entropy_ = self.entropy_
        result.objective_function_ = self.objective_function()
        
        self.before_time_ = time()
        
        self.result_list_.append(result)
        
        
    def save_record(self, save_option= False, filename= None):
        
        '''
            Function that makes pandas dataframe with each iteration 
            information.
        '''
        
        t, c, time, objective_function = [], [], [], []
        
        for result in self.result_list_:
            t.append(result.iteration_)
            c.append(result.c_)
            time.append(result.time_)
            objective_function.append(result.objective_function_)
        
        df = {}
        df['iteration'] = t
        df['c'] = c
        df['time'] = time
        df['objective_function'] = objective_function
        df= pd.DataFrame(df, columns= [
            'iteration', 'c', 'time', 'objective_function'])
        
        if save_option:
            df.to_csv('../result/%s.csv'%filename, index= False, sep= ',')

        return df
        
        
    def objective_function(self):
        
        '''
            Calculate objective function, negative log likelihood function 
            with iteration information.
        '''
        
        likelihood = np.zeros((self.n_, self.c_))
                        
        for i in range(self.c_):
            
            likelihood[:, i] = multivariate_normal(
                self.means_[i], self.covs_[i]).pdf(self.X_)
            
        likelihood = likelihood * self.pi_
        resposibility = self.predict_proba(self.X_)
        
        log_likelihood = \
            np.sum(np.log(
                likelihood + self.smoothing_parameter_) * resposibility) \
            + self.beta_ * self.entropy_ * self.n_
        
        return log_likelihood
    

        
if __name__== '__main__':
        
    pass