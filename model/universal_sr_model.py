import torch
import torch.nn as nn

from .trunk import ResNet
from .polling import SAP, TAP


class UniversalSRModel(nn.Module):
    """Universal Speaker Recognition Model

    Args:
        trunk_net: trunk net type: resnet
        polling_net: polling net type: sap, tap
        num_filterbanks: number of filterbank in transform
        num_frames: number of timesteps in minibatch
        repr_dim: dimension of speaker representation

        layers (resnet specific param): list of block in resnet layer

    """
    def __init__(self, **kwargs):
        super(UniversalSRModel, self).__init__()
        trunk_net = kwargs['trunk_net']
        polling_net = kwargs['polling_net']

        # feat norm
        self.instancenorm = nn.InstanceNorm2d(kwargs['num_filterbanks'])

        # trunk network
        if trunk_net == 'resnet':
            self.trunk = ResNet(layers=kwargs['layers'])
        else:
            raise ValueError('select a valid trunk network')

        # prob trunk network
        pooling_args = self.prob_trunk_network(
            kwargs['num_filterbanks'],
            kwargs['num_frames']
        )

        # pooling network
        if polling_net == 'sap':
            self.poll = SAP(*pooling_args)
        elif polling_net == 'tap':
            self.poll = TAP(*pooling_args)
        else:
            raise ValueError('select a valid polling network')

        # representation layer
        self.repr_layer = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(self.poll.hid_dim, kwargs['repr_dim'])
        )

    def prob_trunk_network(self, freqs, timesteps):
        N, C = 4, 1
        with torch.no_grad():
            x = torch.rand((N, C, freqs, timesteps))
            x = self.trunk(x)
            N, C, F, T = x.size()
        return C, F, T

    def forward(self, x):
        """forward pass

        Args:
            x: tensor of the shape: Nx1xFxT

        Returns:
            tensor of the shape: Nx(repr_dim)
        """
        x = self.instancenorm(x)
        x = self.trunk(x)
        x = self.poll(x)
        x = self.repr_layer(x)
        return x
