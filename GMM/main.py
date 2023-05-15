import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as mvn
from scipy import random

from typing import List, Optional, Tuple, Union

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from tqdm import tqdm


class GMM:

	def __init__(self, k: int, dim: int, init_mu=None, init_cov=None, init_pi=None, colors=None):
		"""Define a Gaussian mixture model with known number of clusters and dimensions.

		Input:
		- k: number of Gaussian clusters.
		- dim: dimension.
		- init_mu: initial values for the means of clusters (k, dim)
				(default) random from uniform [-10, 10]
		- init_cov initial values for the covariance matrix of clusters (k, dim, dim)
				(default) identity matrix for each cluster
		- init_pi: initial values for cluster weights (k,)
				(default) equal value for all cluster, i.e. 1/k
		- colors: color values for plotting each cluster (k, 3)
				(default) random from uniform[0, 1]
		"""

		self.k = k
		self.dim = dim

		if init_mu is None:
			init_mu = random.rand(k, dim) * 20 - 10
		self.mu = init_mu

		if init_cov is None:
			init_cov = np.zeros((k, dim, dim))   # create 3D covariance matrices for all clusters
			for i in range(k):
				init_cov[i] = np.eye(dim)        # define each covariance for all clusters to be identity
		self.cov = init_cov

		if init_pi is None:
			init_pi = np.ones(self.k) / self.k     # uniform distribution for each prior (latent)
		self.pi = init_pi

		if colors is None:
			colors = random.rand(k, 3)
		self.colors = colors

	def _init_em(self, X):
		"""Initialization for EM algorithm.

		- X: data (batch_size, dim)
		"""

		self.X = X
		self.num_points = X.shape[0]
		self.z = np.zeros((self.num_points, self.k))  # initialize responsibilities matrix

	def _e_step(self):
		"""E-step of EM algorithm.

		Evaluate the responsibilities using the current parameter values.
		"""

		for i in range(self.k):
			self.z[:, i] = self.pi[i] * mvn.pdf(self.X, mean=self.mu[i], cov=self.cov[i])

		self.z /= self.z.sum(axis=1, keepdims=True)     # compute the responsibilities for all data points

	def _m_step(self):
		"""M-step of EM algorithm.

		Re-estimate the parameters using the current responsibilities.
		"""

		sum_z = self.z.sum(axis=0)                      # compute all N_k
		self.pi = sum_z / self.num_points               # compute new prior (latent)

		# update mu
		self.mu = np.matmul(self.z.T, self.X)
		self.mu /= sum_z[:, None]

		# compute new covariance matrices
		for i in range(self.k):
			j = np.expand_dims(self.X, axis=1) - self.mu[i]
			s = np.matmul(j.transpose([0, 2, 1]), j)
			self.cov[i] = np.matmul(s.transpose([1, 2, 0]), self.z[:, i])
			self.cov[i] /= sum_z[i]

	def _log_likelihood(self, X):
		"""Compute the log-likelihood of X under current parameters.
		Input:
		- X: data (batch_size, dim)
		Output:
		- log-likelihood of X: Sum_n log(Sum_k (pi_k * N(X_n | mu_k, sigma_k)))
		"""

		ll = []
		for x in X:
			margin = 0
			# compute marginal distribution of x over all latent variables z
			for i in range(self.k):
				margin += self.pi[i] * mvn.pdf(x, mean=self.mu[i], cov=self.cov[i])

			ll.append(np.log(margin))

		return np.sum(ll)

	def fit(self, X, max_iter=10_000, tol=1e-4):
		"""Fit GMM to training data.
		"""

		self._init_em(X)
		ll = []
		ll.append(self._log_likelihood(X))
		self._e_step()
		self._m_step()
		ll.append(self._log_likelihood(X))

		for iter in tqdm(range(max_iter)):
			if ll[-1] - ll[-2] > tol:
				self._e_step()
				self._m_step()
				ll.append(self._log_likelihood(X))
			else:
				print("Terminate at {}-th iteration. Log-likelihood is {}".format(iter + 1, ll[-1]))
				plt.plot(ll, marker='.')
				plt.show()
				break

	def plot_gaussian(self, mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
		"""Utility function to plot one Gaussian from mean and covariance.
		"""

		pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
		ell_radius_x = np.sqrt(1 + pearson)
		ell_radius_y = np.sqrt(1 - pearson)

		ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

		scale_x = np.sqrt(cov[0, 0]) * n_std
		mean_x = mean[0]
		scale_y = np.sqrt(cov[1, 1]) * n_std
		mean_y = mean[1]

		transform = transforms.Affine2D() \
			.rotate_deg(45) \
			.scale(scale_x, scale_y) \
			.translate(mean_x, mean_y)

		ellipse.set_transform(transform + ax.transData)

		return ax.add_patch(ellipse)

	def draw(self, ax, n_std=2.0, facecolor='none', **kwargs):
		"""Function to draw the Gaussians for 2D dataset.
		"""

		if self.dim != 2:
			print("Drawing is available only for 2D case.")
			return

		for i in range(self.k):
			self.plot_gaussian(self.mu[i], self.cov[i], ax, n_std, edgecolor=self.colors[i], **kwargs)

















