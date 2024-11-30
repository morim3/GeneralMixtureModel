from typing import List, Union

import torch
from torch import nn, optim
from torch.distributions import Distribution
from tqdm import tqdm
import numpy as np
from scipy.special import digamma
from sklearn.cluster import KMeans


class GeneralizedMixtureModel(nn.Module):
    def __init__(self,
                 distributions: List[nn.Module],
                 max_iter=1000,
                 rtol=1e-8,
                 random_state=123,
                 init_cluster_ratio=None,
                 maximization_step=100,
                 learning_rate=0.1):

        super().__init__()

        self.distributions = nn.ModuleList(distributions)
        self.max_iter = max_iter
        self.rtol = rtol
        torch.manual_seed(random_state)
        self.maximization_step = maximization_step

        self.cluster_num = len(distributions)
        if init_cluster_ratio is None:
            self.cluster_ratio = torch.ones(self.cluster_num) / self.cluster_num
        else:
            self.cluster_ratio = init_cluster_ratio / init_cluster_ratio.sum()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def fit(self, data):
        prev_log_prob = self.log_prob(data)

        with tqdm(range(self.max_iter)) as pbar:
            for _iter in pbar:
                posterior = self.expectation(data)
                self.maximization(data, posterior)
                log_prob = self.log_prob(data)
                if abs(log_prob - prev_log_prob) / abs(prev_log_prob) < self.rtol:
                    break
                pbar.set_postfix({"log_prob": log_prob})
                prev_log_prob = log_prob

    def log_prob(self, data):
        with torch.no_grad():
            prob = torch.stack([dist.log_prob(data).exp() for dist in self.distributions])
            mixture_log_prob = self.cluster_ratio.matmul(prob).log().sum()
        return mixture_log_prob

    def predict(self, data):
        return self.expectation(data)

    def fit_predict(self, data):
        self.fit(data)
        return self.predict(data)

    def expectation(self, data):
        with torch.no_grad():
            prob = torch.stack([dist.log_prob(data).exp() for dist in self.distributions])
            posterior = self.cluster_ratio.unsqueeze(-1) * prob
            return posterior / posterior.sum(dim=0)

    def maximization(self, data, posterior):
        eps = 1e-7
        self.cluster_ratio = posterior.mean(dim=1)

        for _step in range(self.maximization_step):
            self.optimizer.zero_grad()
            log_prob = torch.stack([dist.log_prob(data) for dist in self.distributions])
            minus_lower_bound = - ((log_prob + self.cluster_ratio.log().unsqueeze(-1)) * posterior).mean()

            loss = minus_lower_bound  # TODO: prior loss
            if torch.isnan(loss):
                raise NotImplementedError

            loss.backward()
            self.optimizer.step()

    def sample(self, n_sample):
        class_sample = torch.multinomial(self.cluster_ratio, n_sample, replacement=True)
        return class_sample, torch.stack([self.distributions[i].sample() for i in class_sample])


class VariationalGaussianMixture:
    def __init__(self, cluster_num: int, dim: int, init_params):

        self.cluster_num = cluster_num
        self.dim = dim

        self.prior_dirichlet = init_params.prior_dirichlet
        self.prior_loc_mu = init_params.prior_loc_mu
        self.prior_beta = init_params.prior_beta
        self.prior_cov_mu = init_params.prior_cov_mu
        self.prior_cov_shape = init_params.prior_cov_shape



    def _maximize(self, data, rho):
        """

        :param data: (batch_size n, dim m) numpy array
        :return:
        """
        # maximization
        N_k = rho.sum(axis=0) + 1e-7
        estimated_mu = np.sum(data[:, np.newaxis, :] * rho[:, :, np.newaxis], axis=0) / N_k[:, np.newaxis]
        variation = data[:, np.newaxis] - estimated_mu[np.newaxis, :]
        estimated_cov = np.sum(rho[:, :, np.newaxis, np.newaxis] *
                                variation[:, :, :, np.newaxis] *
                                variation[:, :, np.newaxis, :], axis=0) / N_k[:, np.newaxis, np.newaxis]

        variation_from_prior = estimated_mu - self.prior_loc_mu
        self.post_dirichlet = self.prior_dirichlet + N_k
        self.post_beta = self.prior_beta + N_k
        self.post_loc_mu = (self.prior_beta * self.prior_loc_mu + N_k[:, np.newaxis] * estimated_mu) / self.post_beta[:, np.newaxis]

        self.post_cov_mu = np.linalg.inv(self.prior_cov_mu) + estimated_cov * N_k[:, np.newaxis, np.newaxis] \
                           + (self.prior_beta * N_k / (self.prior_beta + N_k))[:, np.newaxis, np.newaxis] \
                           * variation_from_prior[:, np.newaxis, :] * variation_from_prior[:, :, np.newaxis]

        self.post_cov_mu = np.linalg.inv(self.post_cov_mu)
        self.post_cov_shape = self.prior_cov_shape + N_k

    def fit(self, data, max_iteration=10):
        rho = np.eye(self.cluster_num)[KMeans(self.cluster_num).fit_predict(data)]
        for i in range(max_iteration):

            self._maximize(data, rho)
            rho = self.predict(data)

    def predict(self, data):
        E_ln_pi = np.array([digamma(a_k) for a_k in self.post_dirichlet]) - digamma(self.post_dirichlet.sum())
        E_ln_det_covar = np.array([np.sum([digamma(self.post_cov_shape[k]+1-i) for i in range(data.shape[1])])
                                   + data.shape[1] * np.log(2)
                                   + np.linalg.slogdet(self.post_cov_mu[k])[1]
                                   for k in range(self.cluster_num)]).transpose()

        deviation = data[:, np.newaxis, :] - self.post_loc_mu[np.newaxis, :, :]

        E_mahalanobis = ((deviation[:, :, :, np.newaxis] * self.post_cov_mu[np.newaxis]).sum(axis=-2) * deviation).sum(axis=-1) \
                        * self.post_cov_shape[np.newaxis, :]

        rho = np.exp(E_ln_pi[np.newaxis, :] + E_ln_det_covar / 2 - data.shape[1] * np.log(2) / 2 - E_mahalanobis / 2)
        rho = rho / np.sum(rho, axis=1)[:, np.newaxis]
        return rho
















