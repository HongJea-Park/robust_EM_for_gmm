# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 06:34:04 2019

@author: hongj
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from generator.sample_generator import generator_multivariate_normal
from robustEM import rEM
from visualization import visualization_2d as vs2
from visualization import plot


# Data Generate
means = [[.0, .0],
         [.0, .0],
         [-1.5, 1.5],
         [1.5, 1.5],
         [0, -2]]
covs = [[[.01, .0], [.0, 1.25]],
        [[8, .0], [.0, 8]],
        [[.2, .0], [.0, .015]],
        [[.2, .0], [.0, .015]],
        [[1., .0], [.0, .2]]]
mix_prob = [.2, .2, .2, .2, .2]

ex5 = generator_multivariate_normal(means=means,
                                    covs=covs,
                                    mix_prob=mix_prob)

X = ex5.get_sample(1000)

# Real
means_real = ex5.means_
covs_real = ex5.covs_

fig, ax = plt.subplots(1, 1)
vs2.scatter_sample(ax, X, 'Real Data and Real Gaussian Distribution')
vs2.get_figure(ax, means_real, covs_real, 'b', 'b')

plt.savefig('../plot/example5_1.png', dpi=300)

# Standard EM with initial values
init_idx = np.random.choice(np.arange(X.shape[0]), 5)
means_init = X[init_idx, :]
gmm_random = GaussianMixture(n_components=5, means_init=means_init)
gmm_random.fit(X)
means_sklearn_random = gmm_random.means_
covs_sklearn_random = gmm_random.covariances_

# Standard EM with Kmeans initial values
gmm_kmeans = GaussianMixture(n_components=5)
gmm_kmeans.fit(X)
means_sklearn_Kmeans = gmm_kmeans.means_
covs_sklearn_Kmeans = gmm_kmeans.covariances_

# visualization with Real, standard EM with initial values,
# standard EM with Kmeans initial values
plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=.2)
ax_list = [plt.subplot(1, 2, i) for i in range(1, 3)]
title_list = ['Standard EM with random initial values',
              'Standard EM with Kmeans initial values']
means_list = [means_sklearn_random, means_sklearn_Kmeans]
covs_list = [covs_sklearn_random, covs_sklearn_Kmeans]

for i in range(2):

    vs2.scatter_sample(ax_list[i], X, title_list[i])
    vs2.get_figure(ax_list[i], means_list[i], covs_list[i],
                   'tab:red', 'tab:red')
    vs2.get_figure(ax_list[i], means_real, covs_real, 'b', 'b', ls=':')
    ax_list[i].legend(handles=[plt.plot([], ls=':', color='b')[0],
                               plt.plot([], ls='-', color='tab:red')[0]],
                      labels=['R', 'E'],
                      loc='lower left',
                      fontsize=7)

plt.savefig('../plot/example5_2.png', dpi=300)

# robust EM
rem = rEM.robustEM()
rem.fit(X)

results = rem.result_list_
record = rem.save_record(save_option=True, filename='example5')

# visualization
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Example5 with robust EM algorithm', fontsize=16)
plt.subplots_adjust(wspace=.3, hspace=.3)

ax1, ax2, ax3, ax4, ax5, ax6 = (plt.subplot(231), plt.subplot(232),
                                plt.subplot(233), plt.subplot(234),
                                plt.subplot(235), plt.subplot(236))

# vs2.scatter_sample(ax1, X, 'Real Data and Real Gaussian Distribution')

ax_list = [ax1, ax2, ax3, ax4, ax5]
idx = [0, 1, 5, 20, -1]

for ax, idx in zip(ax_list, idx_):
    result = results[idx]
    vs2.scatter_sample(
        ax, X, 'Iteration: %s; C: %s' % (result.iteration_, result.c_))
    vs2.get_figure(ax, means_real, covs_real, 'b', 'b', ls=':')
    vs2.get_figure(ax, result.means_, result.covs_, 'tab:red', 'tab:red')
    ax.legend(handles=[plt.plot([], ls=':', color='b')[0],
                       plt.plot([], ls='-', color='tab:red')[0]],
              labels=['R', 'E'],
              loc='lower left',
              fontsize=7)

plot.objective_function_plot(ax6, results, 'darkorange')
plt.savefig('../plot/example5_3.png', dpi=300)
