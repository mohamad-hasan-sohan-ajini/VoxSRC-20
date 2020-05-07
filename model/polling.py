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

        self.hid_dim = channels * freqs
        self.timesteps = timesteps

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
        x = x.view(-1, self.hid_dim, self.timesteps)
        x = x.mean(dim=2)
        return x


class SAP(Polling):
    """Self Attentive Polling"""
    def __init__(self, channels, freqs, timesteps):
        super(SAP, self).__init__(channels, freqs, timesteps)

        self.main = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hid_dim // 2, 1)
        )
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        y = x.view(-1, self.hid_dim, self.timesteps)
        x = y.transpose(1, 2).contiguous().view(-1, self.hid_dim)
        x = self.main(x)
        x = x.view(-1, self.timesteps, 1)
        x = torch.matmul(y, x).view(-1, self.hid_dim)
        return x
