# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:17:50 2019

@author: hongj
"""

import numpy as np
from scipy.stats import norm


def histogram(ax, X, title):
    
    '''
        Histogram function. 
        
    Args: 
        ax: matplotlib.axes.Axes
        X: numpy array
        title: string
    '''

    
    ax.set_title(title)
    ax.hist(X, bins= int(X.shape[0]/ 50), color= 'g', histtype= u'barstacked', density= False)
    ax.set_xlim(np.min(X)- 1, np.max(X)+ 1)
        

def curve(ax, X, means, covs, mix_prob, color, label):
    
    '''
        Function that plots pdf of gaussian mixture.
        
    Args:
        ax: matplotlib.axes.Axes
        X: numpy array
        means: numpy array
        covs: numpy array
        mix_prob: numpy array or list
        color: string
        label: string
    '''
        
    X_range= np.linspace(X.min()- 1, X.max()+ 1, 200).reshape(-1, 1)
    
    c= len(means)
    
    prob= np.zeros((X_range.shape[0], c))
    
    for i in range(c):
        
        prob[:, i]= norm(means[i], covs[i]).pdf(X_range).flatten()
        
    prob= (prob* mix_prob).sum(axis= 1)
    
    ax.plot(X_range, prob, c= color, linewidth= 1.5, label= label)
    

def get_figure(ax, X, means, covs, mix_prob, result, title):
    
    '''
        Function for convenient visualization.
        
    Args:
        ax: matplotlib.axes.Axes
        X: numpy array
        means: numpy array
        covs: numpy array
        mix_prob: numpy array or list
        result: Class for recording iteration results in robustEM.rEM
        title: string
    '''

    ax.set_title(title)

    curve(ax, X, means, covs, mix_prob, 'b', label= 'Real Model')
    curve(ax, X, result.means_, np.sqrt(result.covs_), result.mix_prob_, \
          'tab:red', label= 'Estimate Model')
    
    ax.legend(loc= 'upper right')
        
        
def objective_function_plot(ax, results, color):
    
    '''
        Function that plots objective function at each step.
        
    Args:
        ax: matplotlib.axes.Axes
        results: list of class for recording iteration results in robustEM.rEM
    '''
    
    x= np.arange(len(results))
    obj= [result.objective_function_ for result in results]
    
    ax.plot(x, obj, c= color, linewidth= 1.5)
    ax.set_xlabel('iteration')
    ax.set_title('Objective Function per Iteration')
    
    
def time_cost_plot(ax, df, title):
    
    '''
        Function that plots computation time cost at each step.
        
    Args:
        ax: matplotlib.axes.Axes
        df: pandas DataFrame
    '''
    
    ax.plot(np.arange(df.shape[0])[1:], df['time'][1:], 'tab:red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time(sec)')
    ax.set_title(title, fontsize= 10)
