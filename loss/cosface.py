import torch
import torch.nn as nn


class CosFace(nn.Module):
    def __init__(self, repr_dim, num_spkr, m, s):
        super(CosFace, self).__init__()

        w = nn.init.xavier_normal_(torch.zeros((repr_dim, num_spkr)), gain=1)
        self.w = nn.Parameter(w, requires_grad=True)
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def normalizer(self, x, dim):
        x = torch.div(x, x.norm(dim=dim, keepdim=True).clamp(1e-6))
        return x

    def forward(self, net_out, label):
        """ calculate CosFace loss
        inputs:
            net_out: of shape Nxrepr_size
            label: of size Nx1
        return:
            scores: cosine similarity to each speaker
            loss: cosine similarity loss value averaged by batch
        """
        # normalize x, w
        net_out = self.normalizer(net_out, 1)
        w = self.normalizer(self.w, 0)
        # cosine values
        cos = torch.matmul(net_out, w)
        # m matrix
        m = torch.zeros_like(cos).scatter(1, label, self.m)
        # scores
        scores = self.s * (cos - m)
        # Softmax + CE
        loss = self.ce(scores, label.view(-1))
        return scores, loss


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    repr_dim = 128
    num_spkr = 10
    m = .1
    s = 15.

    criterion = CosFace(repr_dim, num_spkr, m, s).to(device)
    x = torch.rand((4, repr_dim)).to(device)
    label = torch.LongTensor([[1], [4], [7], [9]]).to(device)
    print(criterion(x, label))
