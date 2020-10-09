import torch
import torch.nn as nn


class Pooling(nn.Module):
    """Pooling layer template class
    reduces the channels, freqs and time axis to represent an utterance
    level vector

    Args:
        channels: num out channels of trunk network
        freqs: num out freqs of trunk network
        timesteps: num out timesteps of trunk network
    """
    def __init__(self, channels, freqs, timesteps):
        super(Pooling, self).__init__()

        self.hid_dim = channels
        self.freqs = freqs

    def forward(self, x):
        """forward pass

        Args:
            x: tensor of the shape: NxCxFxT

        Returns:
            tensor of the shape: Nx(hid_dim)
        """
        raise NotImplementedError


class TAP(Pooling):
    """Temporal Averaging Pooling"""
    def __init__(self, channels, freqs, timesteps):
        super(TAP, self).__init__(channels, freqs, timesteps)

    def forward(self, x):
        x = x.mean(dim=2).mean(dim=2)
        return x


class SAP(Pooling):
    """Self Attentive Pooling"""
    def __init__(self, channels, freqs, timesteps):
        super(SAP, self).__init__(channels, freqs, timesteps)

        self.sap_linear = nn.Linear(channels, channels)
        self.attention = nn.Parameter(
            nn.init.xavier_normal_(torch.zeros(channels, 1))
        )
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.mean(dim=2).transpose(1, 2).contiguous()
        h = torch.tanh(self.sap_linear(x))
        w = torch.matmul(h, self.attention)
        w = self.sm(w)
        x = torch.sum(x * w, dim=1)
        return x


class SAP2(Pooling):
    """Self Attentive Pooling + weighter reduction"""
    def __init__(self, channels, freqs, timesteps):
        super(SAP2, self).__init__(channels, freqs, timesteps)

        self.freq_reduction = nn.Parameter(
            nn.init.xavier_normal_(torch.zeros((freqs, 1)), gain=1)
        )
        self.sap_linear = nn.Linear(channels, channels)
        self.attention = nn.Parameter(
            nn.init.xavier_normal_(torch.zeros(channels, 1))
        )
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        # wighted freq reduction
        x = x.transpose(2, 3).contiguous()
        x = torch.matmul(x, self.freq_reduction).squeeze(3)
        x = x.transpose(1, 2).contiguous()

        # self attention
        h = torch.tanh(self.sap_linear(x))
        w = torch.matmul(h, self.attention)
        w = self.sm(w)
        x = torch.sum(x * w, dim=1)
        return x
