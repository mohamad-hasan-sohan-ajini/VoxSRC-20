import torch
import torch.nn as nn

from .trunk_resnet import ResNet
from .trunk_resnetse import ResNetSE
from .trunk_tds import TDSModel
from .pooling import SAP, TAP


class UniversalSRModel(nn.Module):
    """Universal Speaker Recognition Model

    Args:
        trunk_net: trunk net type: resnet
        pooling_net: pooling net type: sap, tap
        num_filterbanks: number of filterbank in transform
        num_frames: number of timesteps in minibatch
        repr_dim: dimension of speaker representation

        layers (resnet specific param): list of block in resnet layer
    """
    def __init__(self, n_feat, **kwargs):
        super(UniversalSRModel, self).__init__()
        trunk_net = kwargs['trunk_net']
        pooling_net = kwargs['pooling_net']

        # trunk network
        if trunk_net == 'resnet':
            print('resnet model instance')
            self.trunk = ResNet(layers=kwargs['layers'])
        elif trunk_net == 'resnetse':
            print('resnet se model instance')
            self.trunk = ResNetSE(layers=kwargs['layers'])
        elif trunk_net == 'tds':
            print('tds model instance')
            self.trunk = TDSModel()
        else:
            raise ValueError('select a valid trunk network')

        # prob trunk network
        pooling_args = self.prob_trunk_network(
            n_feat,
            kwargs['n_frames']
        )

        # pooling network
        if pooling_net == 'sap':
            self.poll = SAP(*pooling_args)
        elif pooling_net == 'tap':
            self.poll = TAP(*pooling_args)
        else:
            raise ValueError('select a valid pooling network')

        # representation layer
        repr_dim = kwargs['repr_dim']
        self.repr_layer = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(self.poll.hid_dim, repr_dim),
            nn.BatchNorm1d(repr_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(.2),
            nn.Linear(repr_dim, repr_dim)
        )

    def prob_trunk_network(self, freqs, timesteps):
        N, C = 1, 1
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
        x = self.trunk(x)
        x = self.poll(x)
        x = self.repr_layer(x)
        return x
