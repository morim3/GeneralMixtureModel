from typing import Dict, Type

import torch
from torch import nn, distributions


class ModuledDistribution(nn.Module):
    def __init__(self, distribution: distributions.Distribution, init_parameters: Dict):
        super().__init__()
        self.distribution = distribution
        self.parameter = nn.ParameterDict(init_parameters)

    def log_prob(self, data):
        eps = 1e-5
        return self.distribution.log_prob(data) + eps

    def sample(self, sample_shape=torch.Size()):
        return self.distribution.sample(sample_shape)


