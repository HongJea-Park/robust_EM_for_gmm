# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 00:12:10 2019

@author: hongj
"""

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np

from generator.sample_generator import generator_multivariate_normal
from robustEM import rEM
from visualization import visualization_2d as vs2


means = [[.5, .5],
         [2.5, .5],
         [4.5, .5],
         [6.5, .5],
         [.5, 2.5],
         [2.5, 2.5],
         [4.5, 2.5],
         [6.5, 2.5],
         [.5, 4.5],
         [2.5, 4.5],
         [4.5, 4.5],
         [6.5, 4.5],
         [.5, 6.5],
         [2.5, 6.5],
         [4.5, 6.5],
         [6.5, 6.5]]

covs = [[[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]],
        [[.1, .0], [.0, .1]]]


mix_prob = [1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16,
            1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16]

ex6 = generator_multivariate_normal(means=means,
                                    covs=covs,
                                    mix_prob=mix_prob)

X = ex6.get_sample(800)

# Real
means_real = ex6.means_
covs_real = ex6.covs_

fig, ax = plt.subplots(1, 1)
vs2.scatter_sample(ax, X, 'Real Data and Real Gaussian Distribution')

plt.savefig('../plot/example6_1.png', dpi=300)

# Standard EM with initial values
init_idx = np.random.choice(np.arange(X.shape[0]), 16)
means_init = X[init_idx, :]
gmm_random = GaussianMixture(n_components=16, means_init=means_init)
gmm_random.fit(X)
means_sklearn_random = gmm_random.means_
covs_sklearn_random = gmm_random.covariances_

# Standard EM with Kmeans initial values
gmm_kmeans = GaussianMixture(n_components=16)
gmm_kmeans.fit(X)
means_sklearn_Kmeans = gmm_kmeans.means_
covs_sklearn_Kmeans = gmm_kmeans.covariances_

# visualization with Real, standard EM with initial values,
# standard EM with Kmeans initial values
plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=.1)
ax_list = [plt.subplot(1, 2, i) for i in range(1, 3)]
title_list = [
    'Standard EM with random initial values; %s iter' % gmm_random.n_iter_,
    'Standard EM with Kmeans initial values; %s iter' % gmm_kmeans.n_iter_]
means_list = [means_sklearn_random, means_sklearn_Kmeans]
covs_list = [covs_sklearn_random, covs_sklearn_Kmeans]

for i in range(2):

    vs2.scatter_sample(ax_list[i], X, title_list[i])
    vs2.get_figure(
        ax_list[i], means_list[i], covs_list[i], 'tab:red', 'tab:red')
    vs2.get_figure(ax_list[i], means_real, covs_real, 'b', 'b', ls=':')
    ax_list[i].legend(handles=[plt.plot([], ls=':', color='b')[0],
                               plt.plot([], ls='-', color='tab:red')[0]],
                      labels=['R', 'E'],
                      loc='lower left',
                      fontsize=7)

plt.savefig('../plot/example6_2.png', dpi=300)

# robustEM
rem = rEM.robustEM()
rem.fit(X)

results = rem.result_list_
record = rem.save_record()

# visualization
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Example6 with robust EM algorithm', fontsize=16)
plt.subplots_adjust(wspace=.2, hspace=.3)

ax_list = [plt.subplot(2, 3, i) for i in range(1, 7)]
idx_ = [0, 1, 10, 20, 30, -1]

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

plt.savefig('../plot/example6_3.png', dpi=300)
