from typing import Dict, Type

import torch
from torch import nn, distributions


class ModuledDistribution(nn.Module):
    def __init__(self, distribution: Type, init_parameters: Dict):
        super().__init__()
        self.distribution = distribution
        self.parameter = nn.ParameterDict(init_parameters)

    def log_prob(self, data):
        return self.distribution(**self.parameter).log_prob(data)

    def sample(self, sample_shape=torch.Size()):
        return self.distribution(**self.parameter).sample(sample_shape)

