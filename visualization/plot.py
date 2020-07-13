# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:33:00 2019

@author: hongj
"""

import numpy as np


def objective_function_plot(ax, results, color):

    '''
        Function that plots objective function at each step.

    Args:
        ax: matplotlib.axes.Axes
        results: list of class for recording iteration results in robustEM.rEM
    '''

    x = np.arange(len(results))
    obj = [result.objective_function_ for result in results]

    ax.plot(x, obj, c=color, linewidth=1.5)
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
    ax.set_title(title, fontsize=10)
