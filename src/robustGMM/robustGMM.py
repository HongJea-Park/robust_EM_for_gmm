from time import time

import numpy as np
from scipy.stats import multivariate_normal


class RobustGMM:
    """
    A robust EM clustering algorithm for Gaussian Mixture Models

    Args:
        gamma:
            float. Non-negative regularization added to the diagonal of
            covariance. This variable is equivalent to 'reg_covar' in
            sklearn.mixture.GaussianMixture.
        eps:
            float. The convergence threshold. This variable is equivalent to
            'tol' in sklearn.mixture.GaussianMixture.
    """
    def __init__(self, gamma=1e-4, eps=1e-3):
        self.gamma = gamma
        self.eps = eps
        self.__smoothing_parameter = 1e-256
        self.__training_info = []

    def fit(self, X: np.ndarray):
        """
        Function for training model with data X by using robust EM algorithm.
        Refer 'Robust EM clustering algorithm' section in the paper page 4 for
        more details.

        Args:
            X: Input data. Data type should be numpy array.
        """
        # initialize variables
        self.X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.c = self.n
        self.pi = np.ones(self.c) / self.c
        self.means = self.X.copy()
        self.__cov_idx = int(np.ceil(np.sqrt(self.c)))
        self.beta = 1
        self.beta_update = True
        self.t = 0
        self.entropy = (self.pi*np.log(self.pi)).sum()
        self.__initialize_covmat()
        self.z = self.predict_proba(self.X)
        self.before_time = time()
        self.__get_iter_info()
        self.t += 1
        self.num_update_c = 0

        # robust EM algorithm
        while True:
            self.means = self.__update_means()
            self.new_pi = self.__update_pi()
            self.__update_beta()
            self.pi = self.new_pi
            self.new_c = self.__update_c()
            if self.new_c == self.c:
                self.num_update_c += 1
            if self.t >= 60 and self.num_update_c == 60:
                self.beta = 0
                self.beta_update = False
            self.c = self.new_c
            self.__update_cov()
            self.z = self.predict_proba(self.X)
            self.new_means = self.__update_means()
            if self.__check_convergence() < self.eps:
                break
            self.__remove_repeated_components()
            self.__get_iter_info()
            self.t += 1
        self.__get_iter_info()

    def predict_proba(self, X):
        """
        Calculate posterior probability of each component given the data.

        Args:
            X: numpy array
        """
        likelihood = np.zeros((self.n, self.c))
        for i in range(self.c):
            self.covs[i] = self.__check_positive_semidefinite(self.covs[i])
            dist = multivariate_normal(mean=self.means[i], cov=self.covs[i])
            likelihood[:, i] = dist.pdf(X)
        numerator = likelihood * self.pi + self.__smoothing_parameter
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        z = numerator / denominator
        return z

    def predict(self, X):
        """
        Predict the labels for the data samples in X.

        Args:
            X: numpy array
        """
        argmax = self.predict_proba(X)
        return argmax.argmax(axis=1)

    def get_training_info(self):
        """
        Save training record to json file.

        Args:
            filepath: Json file name.
        """
        return self.__training_info

    def __initialize_covmat(self):
        """
        Covariance matrix initialize function.
        """
        D_mat = np.sqrt(np.sum((self.X[None, :]-self.X[:, None])**2, -1))
        self.covs = np.apply_along_axis(
            func1d=lambda x: self.__initialize_covmat_1d(x),
            axis=1, arr=D_mat)
        D_mat_reshape = D_mat.reshape(-1, 1)
        d_min = D_mat_reshape[D_mat_reshape > 0].min()
        self.Q = d_min*np.identity(self.dim)

    def __initialize_covmat_1d(self, d_k):
        """
        self.__initialize_covmat() that uses np.apply_along_axis().
        This function is refered term 27 in the paper.

        Args:
            d_k: numpy 1d array
        """
        d_k = d_k.copy()
        d_k.sort()
        d_k = d_k[d_k != 0]
        return ((d_k[self.__cov_idx] ** 2) * np.identity(self.dim))

    def __update_means(self):
        """
        Mean vectors update step.
        This function is refered term 25 in the paper.
        """
        means_list = []
        for i in range(self.c):
            z = self.z[:, i]
            means_list.append((self.X*z.reshape(-1, 1)).sum(axis=0) / z.sum())
        return np.array(means_list)

    def __update_pi(self):
        """
        Mixing proportions update step.
        This function is refered term 13 in the paper.
        """
        self.pi_EM_ = self.z.sum(axis=0) / self.n
        self.entropy = (self.pi*np.log(self.pi)).sum()
        return self.pi_EM_ + self.beta*self.pi*(np.log(self.pi)-self.entropy)

    def __update_beta(self):
        """
        Beta update step.
        This function is refered term 24 in the paper.
        """
        if self.beta_update:
            self.beta = np.min([self.__left_term_of_beta(),
                                self.__right_term_of_beta()])

    def __left_term_of_beta(self):
        """
        Left term of beta update step.
        This function is refered term 22 in the paper.
        """
        power = np.trunc(self.dim / 2 - 1)
        eta = np.min([1, 0.5 ** (power)])
        return np.exp(-eta*self.n*np.abs(self.new_pi-self.pi)).sum() / self.c

    def __right_term_of_beta(self):
        """
        Right term of beta update step.
        This function is refered term 23 in the paper.
        """
        pi_EM = np.max(self.pi_EM_)
        pi_old = np.max(self.pi)
        return (1 - pi_EM) / (-pi_old * self.entropy)

    def __update_c(self):
        """
        Update the number of components.
        This function is refered term 14, 15 and 16 in the paper.
        """
        idx_bool = self.pi >= 1 / self.n
        new_c = idx_bool.sum()
        pi = self.pi[idx_bool]
        self.pi = pi / pi.sum()
        z = self.z[:, idx_bool]
        self.z = z / z.sum(axis=1).reshape(-1, 1)
        self.means = self.means[idx_bool, :]
        return new_c

    def __update_cov(self):
        """
        Covariance matrix update step.
        This function is refered term 26 and 28 in the paper.
        """
        cov_list = []
        for i in range(self.new_c):
            new_cov = np.cov((self.X-self.means[i, :]).T,
                             aweights=(self.z[:, i]/self.z[:, i].sum()))
            new_cov = (1-self.gamma)*new_cov-self.gamma*self.Q
            cov_list.append(new_cov)
        self.covs = np.array(cov_list)

    def __check_convergence(self):
        """
        Check whether algorithm converge or not.
        """
        check = np.max(np.sqrt(np.sum((self.new_means-self.means)**2, axis=1)))
        self.means = self.new_means
        return check

    def __check_positive_semidefinite(self, cov):
        """
        Prevent error that covariance matrix is not positive semi definite.
        """
        min_eig = np.min(np.linalg.eigvals(cov))
        if min_eig < 0:
            cov -= 10 * min_eig * np.eye(*cov.shape)
        return cov

    def __get_iter_info(self):
        """
        Record useful information in each step
        for visualization and objective function.
        """
        result = {}
        result['means'] = self.means
        result['covs'] = self.covs
        result['iteration'] = self.t
        result['c'] = self.c
        result['time'] = time() - self.before_time
        result['mix_prob'] = self.pi
        result['beta'] = self.beta
        result['entropy'] = self.entropy
        result['objective_function'] = self.__objective_function()
        self.before_time = time()
        self.__training_info.append(result)

    def __objective_function(self):

        """
        Calculate objective function(negative log likelihood).
        """
        likelihood = np.zeros((self.n, self.c))
        for i in range(self.c):
            likelihood[:, i] = multivariate_normal(
                self.means[i], self.covs[i]).pdf(self.X)
        likelihood = likelihood * self.pi
        resposibility = self.predict_proba(self.X)
        log_likelihood = \
            np.sum(
                np.log(likelihood+self.__smoothing_parameter)*resposibility) \
            + self.beta * self.entropy * self.n
        return log_likelihood

    def __remove_repeated_components(self):
        """
        To remove repeated components during fitting for preventing the
        cases that contain duplicated data.
        """
        c_params = np.concatenate([self.means,
                                   self.covs.reshape(self.c, -1),
                                   self.pi.reshape(-1, 1)],
                                  axis=1)
        _, idx, counts = np.unique(c_params,
                                   axis=0,
                                   return_index=True,
                                   return_counts=True)
        self.means = self.means[idx]
        self.covs = self.covs[idx]
        self.pi = self.pi[idx]*counts
        self.c = self.pi.shape[0]
        self.z = self.z[:, idx]*counts
