import torch
import torch.nn as nn


class Triplet(nn.Module):
    def __init__(self, pos_margine, neg_margine):
        self.pos_margine = pos_margine
        self.neg_margine = neg_margine

    def forward(self, anchor, utterances):
        centroids = torch.stack(utterances, -1).mean(-1)

        pos_distances = nn.functional.pairwise_distance(anchor, centroids)
        # TODO: hard negative
