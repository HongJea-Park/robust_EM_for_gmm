import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np

import robustEM
from data.generator import Generator_Multivariate_Normal
from visualization import visualization_2d as vs2

# Data Generate
means = [[-4, -4], [-4, -4], [2, 2], [-1, -6]]
covs = [[[1, .5], [.5, 1]],
        [[6, -2], [-2, 6]],
        [[2, -1], [-1, 2]],
        [[.125, .0], [.0, .125]]]
mix_prob = [.3, .3, .3, .1]

ex3 = Generator_Multivariate_Normal(means=means,
                                    covs=covs,
                                    mix_prob=mix_prob)

X = ex3.get_sample(1000)

# Real
means_real = ex3.means_
covs_real = ex3.covs_

fig, ax = plt.subplots(1, 1)
vs2.scatter_sample(ax, X, 'Real Data and Real Gaussian Distribution')
vs2.get_figure(ax, means_real, covs_real, 'b', 'b')

plt.savefig('../plot/example3_1.png', dpi=300)

# Standard EM with initial values
init_idx = np.random.choice(np.arange(X.shape[0]), 4)
means_init = X[init_idx, :]
gmm_random = GaussianMixture(n_components=4, means_init=means_init)
gmm_random.fit(X)
means_sklearn_random = gmm_random.means_
covs_sklearn_random = gmm_random.covariances_

# Standard EM with Kmeans initial values
gmm_kmeans = GaussianMixture(n_components=4)
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

plt.savefig('../plot/example3_2.png', dpi=300)

# robust EM
rem = robustEM.RobustEM()
rem.fit(X)

results = rem.result_list_
record = rem.save_record()

# visualization
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Example3 with robust EM algorithm', fontsize=16)
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

plt.savefig('../plot/example3_3.png', dpi=300)

# Experiments by generating 100 data sets with same parameter
# c_list = []

# for _ in range(100):

#     ex1 = generator_multivariate_normal(means=means,
#                                         covs=covs,
#                                         mix_prob=mix_prob)

#     X = ex1.get_sample(1000)

#     rem = rEM.robustEM()
#     rem.fit(X)

#     c_list.append(rem.c_ == 4)

# the number of correct robustEM: 62
