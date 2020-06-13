import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Normalize data which is in the shape of NxCxHxT
    Normalization includes time (like BatchNorm) and excludes batch (like
    LayerNorm).
    """
    def __init__(
        self,
        feat_size,
        feat_dim=1,
        reduced_dims=[2, 3],
        eps=1e-6,
        affine=True
    ):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.feat_dim = feat_dim
        self.reduced_dims = reduced_dims
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(feat_size))
            self.beta = nn.Parameter(torch.zeros(feat_size))

    def forward(self, x):
        mean = x.mean(self.reduced_dims, keepdim=True)
        std = x.std(self.reduced_dims, keepdim=True)
        norm = (x - mean) / (std + self.eps)
        if self.affine:
            norm = norm.transpose(self.feat_dim, -1).contiguous()
            norm = self.gamma * norm + self.beta
            norm = norm.transpose(self.feat_dim, -1).contiguous()
        return norm


class TimeConvBlock(nn.Module):
    def __init__(self, in_channels, time_kernel_size):
        super(TimeConvBlock, self).__init__()

        self.time_conv = nn.Conv2d(
            in_channels,
            in_channels,
            (1, time_kernel_size),
            padding=(0, time_kernel_size // 2)
        )
        self.non_lin = nn.LeakyReLU(inplace=True)
        self.channel_norm = LayerNorm(in_channels, 1, [2, 3])

    def forward(self, x):
        # x: NxCxFxT
        y = self.time_conv(x)
        y = self.non_lin(y)
        y = x + y
        y = self.channel_norm(y)
        return y


class FCBlock(nn.Module):
    def __init__(self, in_channels, freq_size, hidden_dim, dropout):
        super(FCBlock, self).__init__()

        self.lin1 = nn.Linear(in_channels * freq_size, hidden_dim)
        self.non_lin = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_dim, in_channels * freq_size)
        self.channel_norm = LayerNorm(in_channels, 1, [2, 3])

    def forward(self, x):
        # x: NxCxFxT
        N, C, F, T = x.size()
        y = x.transpose(1, 3).contiguous().view(N * T, F * C)
        y = self.lin1(y)
        y = self.drop(y)
        y = self.non_lin(y)
        y = self.lin2(y)
        y = self.drop(y)
        y = y.view(N, T, F, C).transpose(1, 3).contiguous()
        y = x + y
        y = self.channel_norm(y)
        return y


class TDSBlock(nn.Module):
    """Time Depth Separable Block

    :param int in_channels: num input (and also output) channels
    :param int time_kernel_size: time kernel length
    :param int freq_size: frequency bins
    :param int dropout: dropout ratio for FC block FF networks
    :param int hidden_dim: num hidden neurons in FC block
    """
    def __init__(
        self,
        in_channels,
        time_kernel_size,
        freq_size,
        dropout,
        hidden_dim
    ):
        super(TDSBlock, self).__init__()

        self.time_block = TimeConvBlock(in_channels, time_kernel_size)
        self.fc_block = FCBlock(in_channels, freq_size, hidden_dim, dropout)

    def forward(self, x):
        x = self.time_block(x)
        x = self.fc_block(x)
        return x


class TDSModel(nn.Module):
    def __init__(
        self,
        channels,
        blocks,
        hidden_dims,
        time_kernel_size,
        freq_size,
        dropout
    ):
        super(TDSModel, self).__init__()

        feat_channels = channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                1,
                feat_channels,
                kernel_size=7,
                stride=(2, 1),
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(feat_channels),
            nn.LeakyReLU(inplace=True)
        )
        freq_size = int((freq_size + 2*3 - 7) / 2) + 1

        ind, ch, bl, hid = 1, channels[0], blocks[0], hidden_dims[0]
        self.tds1 = nn.Sequential()
        for i in range(bl):
            self.tds1.add_module(
                f'tds{ind}_{i}',
                TDSBlock(
                    ch,
                    time_kernel_size,
                    freq_size,
                    dropout,
                    hid
                )
            )

        feat_channels = channels[1]
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                ch,
                feat_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(feat_channels),
            nn.LeakyReLU(inplace=True)
        )
        freq_size = int((freq_size + 2*3 - 7) / 2) + 1
        print(f'conv2 channels: {feat_channels}')

        ind, ch, bl, hid = 2, channels[1], blocks[1], hidden_dims[1]
        print(f'ch: {ch}')
        self.tds2 = nn.Sequential()
        for i in range(bl):
            self.tds2.add_module(
                f'tds{ind}_{i}',
                TDSBlock(
                    ch,
                    time_kernel_size,
                    freq_size,
                    dropout,
                    hid
                )
            )

        feat_channels = channels[2]
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                ch,
                feat_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(feat_channels),
            nn.LeakyReLU(inplace=True)
        )
        freq_size = int((freq_size + 2*3 - 7) / 2) + 1

        ind, ch, bl, hid = 3, channels[2], blocks[2], hidden_dims[2]
        self.tds3 = nn.Sequential()
        for i in range(bl):
            self.tds3.add_module(
                f'tds{ind}_{i}',
                TDSBlock(
                    ch,
                    time_kernel_size,
                    freq_size,
                    dropout,
                    hid
                )
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.tds1(x)

        x = self.conv2(x)
        x = self.tds2(x)

        x = self.conv3(x)
        x = self.tds3(x)
        return x


if __name__ == '__main__':
    # TDS block
    print('@TDS BLOCK TEST')
    x = torch.rand((4, 10, 80, 1112))
    print(f'x size: {x.size()}')
    model = TDSBlock(10, 21, 80, .2, 2400)
    y = model(x)
    print(f'y size: {y.size()}')

    # TDS Model
    print('@TDS MODEL TEST')
    x = torch.rand((4, 1, 40, 200))
    print(f'x size: {x.size()}')
    model = TDSModel(
        (10, 12, 14),
        (4, 5, 6),
        (200, 400, 600),
        19,
        40,
        .15
    )
    y = model(x)
    print(f'y size: {y.size()}')
