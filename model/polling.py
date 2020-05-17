import torch
import torch.nn as nn


class Polling(nn.Module):
    """Polling layer template class
    reduces the channels, freqs and time axis to represent an utterance
    level vector

    Args:
        channels: num out channels of trunk network
        freqs: num out freqs of trunk network
        timesteps: num out timesteps of trunk network
    """
    def __init__(self, channels, freqs, timesteps):
        super(Polling, self).__init__()

        self.hid_dim = channels

    def forward(self, x):
        """forward pass

        Args:
            x: tensor of the shape: NxCxFxT

        Returns:
            tensor of the shape: Nx(hid_dim)
        """
        raise NotImplementedError


class TAP(Polling):
    """Temporal Averaging Polling"""
    def __init__(self, channels, freqs, timesteps):
        super(TAP, self).__init__(channels, freqs, timesteps)

    def forward(self, x):
        x = x.mean(dim=2).mean(dim=2)
        return x


class SAP(Polling):
    """Self Attentive Polling"""
    def __init__(self, channels, freqs, timesteps):
        super(SAP, self).__init__(channels, freqs, timesteps)

        self.sap_linear = nn.Linear(channels, channels)
        w = nn.init.xavier_normal(torch.zeros(channels, 1))
        self.attention = nn.Parameter(w)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.mean(dim=2).transpose(1, 2)
        h = torch.tanh(self.sap_linear(x))
        w = torch.matmul(h, self.attention)
        w = self.sm(w)
        x = torch.sum(x * w, dim=1)
        return x
