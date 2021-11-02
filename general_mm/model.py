from typing import List, Union

import torch
from torch import nn, optim
from torch.distributions import Distribution
from tqdm import tqdm


class GeneralizedMixtureModel(nn.Module):
    def __init__(self,
                 distributions: List[nn.Module],
                 max_iter=1000,
                 rtol=1e-8,
                 random_state=123,
                 init_cluster_ratio=None,
                 maximization_step=100,
                 learning_rate=0.001):

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
            minus_lower_bound = - ((log_prob + (self.cluster_ratio+eps).log().unsqueeze(-1)) * posterior).mean()

            loss = minus_lower_bound  # TODO: prior loss
            if torch.isnan(loss):
                raise NotImplementedError

            loss.backward()
            self.optimizer.step()

    def sample(self, n_sample):
        class_sample = torch.multinomial(self.cluster_ratio, n_sample, replacement=True)
        return class_sample, torch.stack([self.distributions[i].sample() for i in class_sample])
