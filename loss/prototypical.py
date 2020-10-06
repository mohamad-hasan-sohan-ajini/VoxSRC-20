import torch
import torch.nn as nn


class Prototypical(nn.Module):
    def __init__(self, repr_dim, num_spkr):
        super(Prototypical, self).__init__()
        self.repr_dim = repr_dim
        self.num_spkr = num_spkr

        w = nn.init.xavier_normal_(torch.zeros((repr_dim, num_spkr)), gain=1)
        self.w = nn.Parameter(w, requires_grad=True)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, net_out, label):
        """ calculate Prototypical loss
        inputs:
            net_out: of shape Nxrepr_size
            label: of size Nx1
        return:
            scores: l2 distance to each speaker
            loss: cosine similarity loss value averaged by batch
        """
        # create the same size
        N = net_out.size(0)
        net_out = net_out.unsqueeze(2).expand(N, self.repr_dim, self.num_spkr)
        w = self.w.unsqueeze(0).expand(N, self.repr_dim, self.num_spkr)
        # scores
        scores = -1 * nn.functional.pairwise_distance(net_out, w)
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

    criterion = Prototypical(repr_dim, num_spkr).to(device)
    x = torch.rand((4, repr_dim)).to(device)
    label = torch.LongTensor([1, 4, 7, 9]).to(device)
    print(criterion(x, label))
