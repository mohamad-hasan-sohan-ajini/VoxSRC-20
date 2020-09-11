import csv
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from tqdm import tqdm
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

from opts import create_argparser
from data_loader import VoxCelebDataset
from model import UniversalSRModel
from utils import save_checkpoint, load_checkpoint


def get_utternace_repr(filepath, repr_cache, model, device, ds):
    if filepath in repr_cache:
        return repr_cache[filepath]

    feats = ds.feature_extractor.load_audio_4test(filepath).to(device)
    timesteps = feats.size(1)
    reprs = []
    for start in torch.linspace(0, timesteps, 4):
        start = start.int().item()
        end = start + ds.feature_extractor.n_frames
        if end < timesteps:
            feat = feats[:, start:end]
            feat.unsqueeze_(0).unsqueeze_(0)
            with torch.no_grad():
                reprs.append(model(feat).cpu().squeeze())
    reprs = torch.stack(reprs).mean(0)
    repr_cache[filepath] = reprs
    return reprs


def cosine_based(rep0, rep1):
    rep0 = rep0 / rep0.norm()
    rep1 = rep1 / rep1.norm()
    return -1 * (rep0 - rep1).norm()


def distance_based(rep0, rep1):
    return -1 * (rep0 - rep1).norm()


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def EER_metric(model, device, args):
    print('-' * 20 + f'EER evaluation' + '-' * 20)
    model.eval()

    # data loader
    ds = VoxCelebDataset(
        args.sample_rate,
        args.win_length,
        args.hop_length,
        args.n_frames,
        args.n_fft,
        args.n_filterbanks,
        args.feat_type,
        'eval',
        args.eval_csv,
        args.samples_per_speaker
    )

    # select similarity measure based on criterion
    if args.criterion in ['cosface', 'psge2e']:
        sim_scorer = cosine_based
    elif args.criterion in ['prototypical']:
        sim_scorer = distance_based

    # calculate socres
    labels, scores, repr_cache = [], [], {}
    for label, filepath0, filepath1 in tqdm(ds):
        repr0 = get_utternace_repr(filepath0, repr_cache, model, device, ds)
        repr1 = get_utternace_repr(filepath1, repr_cache, model, device, ds)
        labels.append(int(label))
        scores.append(sim_scorer(repr0, repr1).item())

    eer = compute_eer(labels, scores)
    model.train()
    return eer


if __name__ == '__main__':
    args = create_argparser().parse_args()
    args.num_spkr = 118
    args.model_path = 'checkpoints/model_1050.pt'
    kwargs = vars(args)

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    feature_extractor = transform(**kwargs).to(device)

    model = UniversalSRModel(**kwargs)
    model.to(device)
    load_checkpoint(model, args.model_path, device)
    model.eval()

    with open(args.test_csv) as f:
        eval_data = list(csv.reader(f, delimiter=' '))

    reps, ids = [], []
    for ind, (label, utt0, utt1) in enumerate(tqdm(eval_data)):
        rep0 = utternace_repr(
            model,
            feature_extractor,
            args.num_frames,
            device,
            utt0
        )
        reps.append(rep0.cpu().numpy())
        ids.append([i for i in utt0.split('/') if i.startswith('id')][0])

        rep1 = utternace_repr(
            model,
            feature_extractor,
            args.num_frames,
            device,
            utt1
        )
        reps.append(rep1.cpu().numpy())
        ids.append([i for i in utt1.split('/') if i.startswith('id')][0])

    id_set = sorted(list(set(ids)))
    id_colors = [np.random.rand(3,) for i in id_set]
    ids_index = [id_set.index(i) for i in ids]

    tsne = TSNE(n_components=2, random_state=0)
    reps_2d = tsne.fit_transform(reps)

    for spk_id, rep in zip(ids_index, reps_2d):
        c = id_colors[spk_id]
        plt.scatter(rep[0], rep[1], c=c)
    plt.savefig('tmp.png', dpi=600)
