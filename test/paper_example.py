import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as transforms
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
# Although Axes3D is not used directly,
# it is imported because it is needed for 3d projection.

from robustGMM import RobustGMM
from robustGMM import Generator_Multivariate_Normal
from robustGMM import Generator_Univariate_Normal


# All functions are for visualization.
def make_ellipses(ax, means, covs, edgecolor, m_color, ls='-', n_std=3):
    def __make_ellipses(mean, cov):
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = mpl.patches.Ellipse((0, 0),
                                      ell_radius_x * 2,
                                      ell_radius_y * 2,
                                      facecolor='none',
                                      edgecolor=edgecolor, lw=1, ls=ls)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y)
        transf = transf.translate(mean[0], mean[1])
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
        ax.add_artist(ellipse)
        ax.set_aspect('equal', 'datalim')
        ax.scatter(mean[0], mean[1], c=m_color, marker='*')
    for mean, cov in zip(means, covs):
        __make_ellipses(mean, cov)


def make_1dplot(ax, X, means, covs, mix_prob, info, title):
    def curve(means, covs, mix_prob, color, label, ls='-'):
        X_range = np.linspace(X.min()-1, X.max()+1, 200).reshape(-1, 1)
        c = len(means)
        prob = np.zeros((X_range.shape[0], c))
        for i in range(c):
            prob[:, i] = norm(means[i], covs[i]).pdf(X_range).flatten()
        prob = (prob * mix_prob).sum(axis=1)
        ax.plot(X_range, prob, c=color, linewidth=1.5, label=label, ls=ls)
    ax.set_title(title)
    curve(means, covs, mix_prob, 'b', label='Real Model', ls=':')
    curve(info['means'], np.sqrt(info['covs']), info['mix_prob'],
          'tab:red', label='Estimate Model')
    ax.legend(loc='upper right')


def objective_function_plot(ax, training_info, color):
    x = np.arange(len(training_info))
    obj = [info['objective_function'] for info in training_info]
    ax.plot(x, obj, c=color, linewidth=1.5)
    ax.set_xlabel('iteration')
    ax.set_title('Objective Function per Iteration')


# %% Example 1
# Generate data from 2 multivariate normal distribution with fixed random seed
np.random.seed(0)
real_means = np.array([[.0, .0], [20, .0]])
real_covs = np.array([[[1, .0], [.0, 1]],
                      [[9, .0], [.0, 9]]])
mix_prob = np.array([.5, .5])
generator = Generator_Multivariate_Normal(means=real_means,
                                          covs=real_covs,
                                          mix_prob=mix_prob)
X = generator.get_sample(800)

# GMM using Standard EM Algorithm with random initial values
init_idx = np.random.choice(np.arange(X.shape[0]), 2)
means_init = X[init_idx, :]
gmm = GaussianMixture(n_components=2, means_init=means_init)
gmm.fit(X)
means_sklearn = gmm.means_
covs_sklearn = gmm.covariances_

# Visualization
plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=.2)
ax1 = plt.subplot(1, 2, 1)
ax1.set_title('Real Data and Real Gaussian Distribution', fontsize=10)
ax1.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
ax1.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
ax1.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
make_ellipses(ax=ax1, means=real_means, covs=real_covs,
              edgecolor='b', m_color='b', ls=':', n_std=3)
ax2 = plt.subplot(1, 2, 2)
ax2.set_title('Standard EM with random initial values', fontsize=10)
ax2.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
ax2.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
ax2.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
make_ellipses(ax=ax2, means=means_sklearn, covs=covs_sklearn,
              edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
plt.suptitle('Example1')
plt.savefig('./figure/example1_1.png', dpi=300)

# GMM using robust EM Algorithm
rgmm = RobustGMM()
rgmm.fit(X)
training_info1 = rgmm.get_training_info()

# Visualization
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Example1 with robust EM algorithm', fontsize=16)
plt.subplots_adjust(wspace=.2, hspace=.3)
axList = [plt.subplot(2, 3, i) for i in range(1, 7)]
idxList = [0, 1, 10, 20, 40, -1]
for ax, idx in zip(axList, idxList):
    info = training_info1[idx]
    title = f"Iteration: {info['iteration']}; C: {info['c']}"
    ax.set_title(title, fontsize=10)
    ax.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
    ax.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
    ax.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
    make_ellipses(ax=ax, means=real_means, covs=real_covs,
                  edgecolor='b', m_color='b', ls=':', n_std=3)
    make_ellipses(ax=ax, means=info['means'], covs=info['covs'],
                  edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
    ax.legend(handles=[plt.plot([], ls=':', color='b')[0],
                       plt.plot([], ls='-', color='tab:red')[0]],
              labels=['R', 'E'],
              loc='lower left',
              fontsize=7)
plt.savefig('./figure/example1_2.png', dpi=300)

# %% Example 2-1
# Generate data from 3 multivariate normal distribution with fixed random seed
np.random.seed(0)
real_means = np.array([[.0, 3], [.0, 5], [.0, 7]])
real_covs = np.array([[[1.2, .0], [.0, .01]],
                      [[1.2, .0], [.0, .01]],
                      [[1.2, .0], [.0, .01]]])
mix_prob = np.array([1/3, 1/3, 1/3])
generator = Generator_Multivariate_Normal(means=real_means,
                                          covs=real_covs,
                                          mix_prob=mix_prob)
X = generator.get_sample(300)

# GMM using Standard EM Algorithm with random initial values
init_idx = np.random.choice(np.arange(X.shape[0]), 3)
means_init = X[init_idx, :]
gmm = GaussianMixture(n_components=3, means_init=means_init)
gmm.fit(X)
means_sklearn = gmm.means_
covs_sklearn = gmm.covariances_

# Visualization
plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=.2)
ax1 = plt.subplot(1, 2, 1)
ax1.set_title('Real Data and Real Gaussian Distribution', fontsize=10)
ax1.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
ax1.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
ax1.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
make_ellipses(ax=ax1, means=real_means, covs=real_covs,
              edgecolor='b', m_color='b', ls=':', n_std=3)
ax2 = plt.subplot(1, 2, 2)
ax2.set_title('Standard EM with random initial values', fontsize=10)
ax2.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
ax2.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
ax2.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
make_ellipses(ax=ax2, means=means_sklearn, covs=covs_sklearn,
              edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
plt.suptitle('Example2-1')
plt.savefig('./figure/example2_1_1.png', dpi=300)

# GMM using robust EM Algorithm
rgmm = RobustGMM()
rgmm.fit(X)
training_info21 = rgmm.get_training_info()

# Visualization
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Example2-1 with robust EM algorithm', fontsize=16)
plt.subplots_adjust(wspace=.2, hspace=.3)
axList = [plt.subplot(2, 3, i) for i in range(1, 7)]
idxList = [0, 1, 5, 10, 20, -1]
for ax, idx in zip(axList, idxList):
    info = training_info21[idx]
    title = f"Iteration: {info['iteration']}; C: {info['c']}"
    ax.set_title(title, fontsize=10)
    ax.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
    ax.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
    ax.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
    make_ellipses(ax=ax, means=real_means, covs=real_covs,
                  edgecolor='b', m_color='b', ls=':', n_std=3)
    make_ellipses(ax=ax, means=info['means'], covs=info['covs'],
                  edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
    ax.legend(handles=[plt.plot([], ls=':', color='b')[0],
                       plt.plot([], ls='-', color='tab:red')[0]],
              labels=['R', 'E'],
              loc='lower left',
              fontsize=7)
plt.savefig('./figure/example2_1_2.png', dpi=300)

# %% Example 2-2
# Generate data from 4 multivariate normal distribution with fixed random seed
np.random.seed(0)
real_means = np.array([[.0, 3], [.0, 5], [.0, 7], [.0, 5]])
real_covs = np.array([[[1.2, .0], [.0, .01]],
                      [[1.2, .0], [.0, .01]],
                      [[1.2, .0], [.0, .01]],
                      [[.01, .0], [.0, .8]]])
mix_prob = np.array([.25, .25, .25, .25])
generator = Generator_Multivariate_Normal(means=real_means,
                                          covs=real_covs,
                                          mix_prob=mix_prob)
X = generator.get_sample(400)

# GMM using Standard EM Algorithm with random initial values
init_idx = np.random.choice(np.arange(X.shape[0]), 4)
means_init = X[init_idx, :]
gmm = GaussianMixture(n_components=4, means_init=means_init)
gmm.fit(X)
means_sklearn = gmm.means_
covs_sklearn = gmm.covariances_

# Visualization
plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=.2)
ax1 = plt.subplot(1, 2, 1)
ax1.set_title('Real Data and Real Gaussian Distribution', fontsize=10)
ax1.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
ax1.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
ax1.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
make_ellipses(ax=ax1, means=real_means, covs=real_covs,
              edgecolor='b', m_color='b', ls=':', n_std=3)
ax2 = plt.subplot(1, 2, 2)
ax2.set_title('Standard EM with random initial values', fontsize=10)
ax2.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
ax2.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
ax2.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
make_ellipses(ax=ax2, means=means_sklearn, covs=covs_sklearn,
              edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
plt.suptitle('Example2-2')
plt.savefig('./figure/example2_2_1.png', dpi=300)

# GMM using robust EM Algorithm
rgmm = RobustGMM()
rgmm.fit(X)
training_info22 = rgmm.get_training_info()

# Visualization
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Example2-2 with robust EM algorithm', fontsize=16)
plt.subplots_adjust(wspace=.2, hspace=.3)
axList = [plt.subplot(2, 3, i) for i in range(1, 7)]
idxList = [0, 1, 5, 10, 20, -1]
for ax, idx in zip(axList, idxList):
    info = training_info22[idx]
    title = f"Iteration: {info['iteration']}; C: {info['c']}"
    ax.set_title(title, fontsize=10)
    ax.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
    ax.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
    ax.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
    make_ellipses(ax=ax, means=real_means, covs=real_covs,
                  edgecolor='b', m_color='b', ls=':', n_std=3)
    make_ellipses(ax=ax, means=info['means'], covs=info['covs'],
                  edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
    ax.legend(handles=[plt.plot([], ls=':', color='b')[0],
                       plt.plot([], ls='-', color='tab:red')[0]],
              labels=['R', 'E'],
              loc='lower left',
              fontsize=7)
plt.savefig('./figure/example2_2_2.png', dpi=300)

# %% Example 3
# Generate data from 4 multivariate normal distribution with fixed random seed
np.random.seed(0)
real_means = np.array([[-4, -4], [-4, -4], [2, 2], [-1, -6]])
real_covs = np.array([[[1, .5], [.5, 1]],
                      [[6, -2], [-2, 6]],
                      [[2, -1], [-1, 2]],
                      [[.125, .0], [.0, .125]]])
mix_prob = np.array([.3, .3, .3, .1])
generator = Generator_Multivariate_Normal(means=real_means,
                                          covs=real_covs,
                                          mix_prob=mix_prob)
X = generator.get_sample(1000)

# GMM using Standard EM Algorithm with random initial values
init_idx = np.random.choice(np.arange(X.shape[0]), 4)
means_init = X[init_idx, :]
gmm = GaussianMixture(n_components=4, means_init=means_init)
gmm.fit(X)
means_sklearn = gmm.means_
covs_sklearn = gmm.covariances_

# Visualization
plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=.2)
ax1 = plt.subplot(1, 2, 1)
ax1.set_title('Real Data and Real Gaussian Distribution', fontsize=10)
ax1.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
ax1.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
ax1.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
make_ellipses(ax=ax1, means=real_means, covs=real_covs,
              edgecolor='b', m_color='b', ls=':', n_std=3)
ax2 = plt.subplot(1, 2, 2)
ax2.set_title('Standard EM with random initial values', fontsize=10)
ax2.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
ax2.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
ax2.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
make_ellipses(ax=ax2, means=means_sklearn, covs=covs_sklearn,
              edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
plt.suptitle('Example3')
plt.savefig('./figure/example3_1.png', dpi=300)

# GMM using robust EM Algorithm
rgmm = RobustGMM()
rgmm.fit(X)
training_info3 = rgmm.get_training_info()

# Visualization
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Example3 with robust EM algorithm', fontsize=16)
plt.subplots_adjust(wspace=.2, hspace=.3)
axList = [plt.subplot(2, 3, i) for i in range(1, 7)]
idxList = [0, 1, 5, 10, 20, -1]
for ax, idx in zip(axList, idxList):
    info = training_info3[idx]
    title = f"Iteration: {info['iteration']}; C: {info['c']}"
    ax.set_title(title, fontsize=10)
    ax.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
    ax.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
    ax.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
    make_ellipses(ax=ax, means=real_means, covs=real_covs,
                  edgecolor='b', m_color='b', ls=':', n_std=3)
    make_ellipses(ax=ax, means=info['means'], covs=info['covs'],
                  edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
    ax.legend(handles=[plt.plot([], ls=':', color='b')[0],
                       plt.plot([], ls='-', color='tab:red')[0]],
              labels=['R', 'E'],
              loc='lower left',
              fontsize=7)
plt.savefig('./figure/example3_2.png', dpi=300)

# %% Example 4
# Generate data from 3 univariate normal distribution with fixed random seed
np.random.seed(42)
real_means = np.array([-11, 0, 13])
real_stds = np.array([2, 4, 3])
mix_prob = np.array([1/3, 1/3, 1/3])
generator = Generator_Univariate_Normal(means=real_means,
                                        stds=real_stds,
                                        mix_prob=mix_prob)
X = generator.get_sample(1000)

# GMM using robust EM Algorithm
rgmm = RobustGMM()
rgmm.fit(X)
training_info4 = rgmm.get_training_info()

# Visualization
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Example4 with robust EM algorithm', fontsize=16)
plt.subplots_adjust(wspace=.3, hspace=.3)
ax1, ax2, ax3, ax4, ax5, ax6 = (plt.subplot(231), plt.subplot(232),
                                plt.subplot(233), plt.subplot(234),
                                plt.subplot(235), plt.subplot(236))
ax1.set_title('Real Data and Real Gaussian Distribution')
ax1.hist(X, bins=int(X.shape[0]/50), color='g',
         histtype=u'barstacked', density=False)
ax1.set_xlim(np.min(X)-1, np.max(X)+1)
axList = [ax2, ax3, ax4, ax5]
idxList = [1, 10, 20, -1]
for ax, idx in zip(axList, idxList):
    info = training_info4[idx]
    title = f"Iteration: {info['iteration']}; C: {info['c']}"
    make_1dplot(ax, X, real_means, real_stds, mix_prob, info, title)
    ax.legend(handles=[plt.plot([], ls=':', color='b')[0],
                       plt.plot([], ls='-', color='tab:red')[0]],
              labels=['R', 'E'],
              loc='upper right',
              fontsize=7)
objective_function_plot(ax6, training_info4, 'darkorange')
plt.savefig('./figure/example4.png', dpi=300)

# %% Example 5
# Generate data from 5 multivariate normal distribution with fixed random seed
np.random.seed(42)
real_means = np.array([[.0, .0],
                       [.0, .0],
                       [-1.5, 1.5],
                       [1.5, 1.5],
                       [0, -2]])
real_covs = np.array([[[.01, .0], [.0, 1.25]],
                      [[8, .0], [.0, 8]],
                      [[.2, .0], [.0, .015]],
                      [[.2, .0], [.0, .015]],
                      [[1., .0], [.0, .2]]])
mix_prob = np.array([.2, .2, .2, .2, .2])
generator = Generator_Multivariate_Normal(means=real_means,
                                          covs=real_covs,
                                          mix_prob=mix_prob)
X = generator.get_sample(800)

# GMM using Standard EM Algorithm with random initial values
init_idx = np.random.choice(np.arange(X.shape[0]), 5)
means_init = X[init_idx, :]
gmm = GaussianMixture(n_components=5, means_init=means_init)
gmm.fit(X)
means_sklearn = gmm.means_
covs_sklearn = gmm.covariances_

# Visualization
plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=.2)
ax1 = plt.subplot(1, 2, 1)
ax1.set_title('Real Data and Real Gaussian Distribution', fontsize=10)
ax1.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
ax1.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
ax1.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
make_ellipses(ax=ax1, means=real_means, covs=real_covs,
              edgecolor='b', m_color='b', ls=':', n_std=3)
ax2 = plt.subplot(1, 2, 2)
ax2.set_title('Standard EM with random initial values', fontsize=10)
ax2.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
ax2.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
ax2.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
make_ellipses(ax=ax2, means=means_sklearn, covs=covs_sklearn,
              edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
plt.suptitle('Example5')
plt.savefig('./figure/example5_1.png', dpi=300)

# GMM using robust EM Algorithm
rgmm = RobustGMM()
rgmm.fit(X)
training_info5 = rgmm.get_training_info()

# Visualization
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Example5 with robust EM algorithm', fontsize=16)
plt.subplots_adjust(wspace=.2, hspace=.3)
axList = [plt.subplot(2, 3, i) for i in range(1, 7)]
idxList = [0, 1, 5, 10, 20, -1]
for ax, idx in zip(axList, idxList):
    info = training_info5[idx]
    title = f"Iteration: {info['iteration']}; C: {info['c']}"
    ax.set_title(title, fontsize=10)
    ax.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
    ax.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
    ax.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
    make_ellipses(ax=ax, means=real_means, covs=real_covs,
                  edgecolor='b', m_color='b', ls=':', n_std=3)
    make_ellipses(ax=ax, means=info['means'], covs=info['covs'],
                  edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
    ax.legend(handles=[plt.plot([], ls=':', color='b')[0],
                       plt.plot([], ls='-', color='tab:red')[0]],
              labels=['R', 'E'],
              loc='lower left',
              fontsize=7)
plt.savefig('./figure/example5_2.png', dpi=300)

# %% Example 6
# Generate data from 5 multivariate normal distribution with fixed random seed
np.random.seed(42)
real_means = np.array([[.5, .5],
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
                       [6.5, 6.5]])
real_covs = np.array([[[.1, .0], [.0, .1]],
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
                      [[.1, .0], [.0, .1]]])
mix_prob = np.array([1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16,
                     1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16])
generator = Generator_Multivariate_Normal(means=real_means,
                                          covs=real_covs,
                                          mix_prob=mix_prob)
X = generator.get_sample(800)

# GMM using Standard EM Algorithm with random initial values
init_idx = np.random.choice(np.arange(X.shape[0]), 16)
means_init = X[init_idx, :]
gmm_random = GaussianMixture(n_components=16, means_init=means_init)
gmm_random.fit(X)
means_sklearn_random = gmm_random.means_
covs_sklearn_random = gmm_random.covariances_

# GMM using Standard EM Algorithm with Kmeans initial values
gmm_kmeans = GaussianMixture(n_components=16)
gmm_kmeans.fit(X)
means_sklearn_Kmeans = gmm_kmeans.means_
covs_sklearn_Kmeans = gmm_kmeans.covariances_

# Visualization
plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=.1)
axList = [plt.subplot(1, 2, i) for i in range(1, 3)]
titleList = [
    f"Standard EM with Random Initial Values",
    f"Standard EM with Kmeans Initial Values"]
meansList = [means_sklearn_random, means_sklearn_Kmeans]
covsList = [covs_sklearn_random, covs_sklearn_Kmeans]
for i in range(2):
    axList[i].set_title(titleList[i], fontsize=10)
    axList[i].scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
    axList[i].set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
    axList[i].set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
    make_ellipses(axList[i], meansList[i], covsList[i], 'tab:red', 'tab:red')
    make_ellipses(axList[i], real_means, real_covs, 'b', 'b', ls=':')
    axList[i].legend(handles=[plt.plot([], ls=':', color='b')[0],
                              plt.plot([], ls='-', color='tab:red')[0]],
                     labels=['R', 'E'],
                     loc='lower left',
                     fontsize=7)
plt.suptitle('Example6')
plt.savefig('./figure/example6_1.png', dpi=300)

# GMM using robust EM Algorithm
rgmm = RobustGMM()
rgmm.fit(X)
training_info6 = rgmm.get_training_info()

# Visualization
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Example3 with robust EM algorithm', fontsize=16)
plt.subplots_adjust(wspace=.2, hspace=.3)
axList = [plt.subplot(2, 3, i) for i in range(1, 7)]
idxList = [0, 1, 5, 10, 20, -1]
for ax, idx in zip(axList, idxList):
    info = training_info6[idx]
    title = f"Iteration: {info['iteration']}; C: {info['c']}"
    ax.set_title(title, fontsize=10)
    ax.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
    ax.set_xlim(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2)
    ax.set_ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2)
    make_ellipses(ax=ax, means=real_means, covs=real_covs,
                  edgecolor='b', m_color='b', ls=':', n_std=3)
    make_ellipses(ax=ax, means=info['means'], covs=info['covs'],
                  edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
    ax.legend(handles=[plt.plot([], ls=':', color='b')[0],
                       plt.plot([], ls='-', color='tab:red')[0]],
              labels=['R', 'E'],
              loc='lower left',
              fontsize=7)
plt.savefig('./figure/example6_2.png', dpi=300)

# %% Example 7
# Generate data from 5 multivariate normal distribution with fixed random seed
np.random.seed(42)
real_means = np.array([[-5, -10, 0],
                       [0, -10, 0],
                       [5, -10, 0],
                       [-5, -0, 4],
                       [0, -0, 4],
                       [5, -0, 4],
                       [-5, 10, 8],
                       [0, 10, 8],
                       [5, 10, 8]])
real_covs = np.array([[[.5, 0, 0], [0, 2, 0], [0, 0, 1]],
                      [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
                      [[.5, 0, 0], [0, 2, 0], [0, 0, 1]],
                      [[.5, 0, 0], [0, 8, 0], [0, 0, 2]],
                      [[1, 0, 0], [0, 8, 0], [0, 0, 2]],
                      [[.5, 0, 0], [0, 8, 0], [0, 0, 2]],
                      [[.5, 0, 0], [0, 2, 0], [0, 0, 1]],
                      [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
                      [[.5, 0, 0], [0, 2, 0], [0, 0, 1]]])
mix_prob = np.array([1/16, 2/16, 1/16, 2/16, 4/16, 2/16, 1/16, 2/16, 1/16])
generator = Generator_Multivariate_Normal(means=real_means,
                                          covs=real_covs,
                                          mix_prob=mix_prob)
X = generator.get_sample(1600)

# GMM using robust EM Algorithm
rgmm = RobustGMM()
rgmm.fit(X)
training_info7 = rgmm.get_training_info()

# Visualization
pred = rgmm.predict(X)
cList = np.unique(pred)
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], color='g')
ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=pred)
ax1.set_title('The 3-dimensional data set with nine clusters')
ax2.set_title('Clustering results by the robust EM algorithm')
plt.suptitle('Example7')
plt.savefig('./figure/example7.png', dpi=300)

# %% Time Cost
trainingList = [training_info4, training_info5, training_info7]
fig = plt.figure(figsize=(12, 5))
fig.suptitle('Computation time cost per each iteration in example 4, 5, 7',
             fontsize=16)
axList = [plt.subplot(131), plt.subplot(132), plt.subplot(133)]
plt.subplots_adjust(wspace=.3)
for ax, training_info, example in zip(axList, trainingList, [4, 5, 7]):
    x = np.arange(len(training_info))
    timeList = [info['time'] for info in training_info]
    ax.plot(x[1:], timeList[1:], 'tab:red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time(sec)')
    ax.set_title(f'Time Cost in {example}', fontsize=10)
plt.savefig('./figure/time_cost.png', dpi=300)
