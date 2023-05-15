import numpy as np
from scipy.special import logsumexp

import typing 
from tqdm import tqdm 

import matplotlib.pyplot as plt


class BMM:
    """EM Algorithm for Bernoulli Mixture Model.
    """

    def __init__(self, k: int, dim: int, alpha=None, init_mu=None, init_pi=None):
        """Define a Bernoulli mixture model with known number of clusters and dimensions.

        Input:
        - k: number of clusters
        - dim: dimension
        - init_theta: initial value for the probability of success
        """

        self.K = k
        self.D = dim
        self.alpha = alpha

        if init_mu is None:
            init_mu = np.random.uniform(.25, .75, (k, dim))
            # init_mu /= np.sum(init_mu, axis=1, keepdims=True)
        self.mu = init_mu

        if init_pi is None:
            init_pi = np.ones(k) / k  # uniform distribution of pi
        self.pi = init_pi

    def _init_em(self, X):
        """Initialization of EM algorithm for Bernoulli mixture model.

        Input:
        - X: data (batch_size, dim)
        """

        self.X = X  # data matrix
        self.N = X.shape[0]  # the number of data points
        self.Z = np.zeros((self.N, self.K))  # responsibility matrix

    def _e_step(self):
        """E-step of EM algorithm.

        Evaluate the responsibilities (Z) using the current parameter values and log-sum-exp.
        """

        for i in range(self.K):
            self.Z[:, i] = np.log(self.pi[i]) + np.sum(
                (self.X * np.log(self.mu[i]) + (1 - self.X) * np.log(1 - self.mu[i])), axis=1)

        self.Z -= logsumexp(self.Z, axis=1, keepdims=True)
        self.Z = np.exp(self.Z)

    def _m_step(self):
        """M-step of EM algorithm.

        Re-estimate the parameters using the current responsibilities Z.
        """

        # update priors
        self.pi = self.Z.sum(axis=0) / self.N

        # update mu using Laplace smoothing
        self.mu = np.matmul(self.Z.T, self.X)
        if self.alpha:
            self.mu += self.alpha
        self.mu /= (self.Z.sum(axis=0) + self.alpha * self.K)[:, np.newaxis]

    def _log_likelihood(self, X):
        """Compute the log-likelihood of X under current parameters.

        Input:
        - X: data (batch_size, dim)

        Output:
        - log-likelihood of X: Sum_n log(Sum_k(pi_k * Ber(x_n | mu_k)))
        """

        ll = 0
        for x in X:
            log_margin = np.log(self.pi)[:, np.newaxis] + np.sum(x * np.log(self.mu) + (1 - x) * np.log(1 - self.mu),
                                                                 axis=1, keepdims=True)
            ll += logsumexp(log_margin)

        return ll

    def fit(self, X, max_iter=100, tol=1e-2):
        """Fit BMM to training data.
        """

        self._init_em(X)
        ll_hist = [self._log_likelihood(X)]

        self._e_step()
        self._m_step()
        ll_hist.append(self._log_likelihood(X))

        for i in tqdm(range(max_iter)):
            if ll_hist[-1] - ll_hist[-2] > tol:
                self._e_step()
                self._m_step()
                ll_hist.append(self._log_likelihood(X))
            else:
                print("Terminate at {}th iteration. Log-likelihood is {}".format(i + 1, ll_hist[-1]))
                plt.plot(ll_hist, marker='.')
                plt.show()
                break
