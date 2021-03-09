import numpy as np


class Generator_Multivariate_Normal():

    '''
        Class for generating n-dimensions sample data set with multivariate
        normal distribution.

    Args:
        means: list
        covs: list
        mix_prob: list
    '''

    def __init__(self, means, covs, mix_prob):

        self.n_components_ = len(means)
        self.dim_ = len(means[0])
        self.means_ = np.array(means)
        self.covs_ = np.array(covs)
        self.mix_prob_ = mix_prob

    def get_sample(self, size):

        '''
            Function that generate sample data sets.

        Args:
            size: integer
        '''

        X = np.zeros((0, self.dim_))
        sample_size = 0

        for n in range(0, self.n_components_):

            mean = self.means_[n]
            cov = self.covs_[n]

            if n != self.n_components_ - 1:
                num_of_sample = int(np.around(size*self.mix_prob_[n]))
                sample_size += num_of_sample
            else:
                num_of_sample = size - sample_size

            X_n = np.random.multivariate_normal(
                mean=mean, cov=cov, size=num_of_sample)
            X = np.vstack([X, X_n])

        return X


class Generator_Univariate_Normal():

    '''
        Class for generating one-dimension sample data set with univariate
        normal distribution.

    Args:
        means: list
        stds: list
        mix_prob: list
    '''

    def __init__(self, means, stds, mix_prob):

        self.n_components_ = len(means)
        self.means_ = np.array(means)
        self.stds_ = np.array(stds)
        self.mix_prob_ = mix_prob

    def get_sample(self, size):

        '''
            Function that generate sample data sets.

        Args:
            size: integer
        '''

        X = np.array([])
        sample_size = 0

        for n in range(0, self.n_components_):

            mean = self.means_[n]
            std = self.stds_[n]

            if n != self.n_components_ - 1:
                num_of_sample = int(np.around(size*self.mix_prob_[n]))
                sample_size += num_of_sample
            else:
                num_of_sample = size - sample_size

            X_n = np.random.normal(loc=mean, scale=std, size=num_of_sample)
            X = np.append(X, X_n)

        return X
