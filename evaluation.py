import csv
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torchaudio
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from tqdm import tqdm
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

from opts import add_args
from data_loader import ClassificationVCDS, MetricLearningVCDS, transform
from model import UniversalSRModel
from utils import save_checkpoint, load_checkpoint


def fix_length(signal, clip_length):
    length = signal.size(1)
    if length == clip_length:
        return signal
    lack = clip_length - length
    pad_left = pad_right = lack // 2
    pad_left += 1 if lack % 2 else 0
    return nn.functional.pad(signal, (pad_left, pad_right))


def utternace_repr(model, transform, num_frames, device, filepath):
    clip_length = (num_frames-3) * transform.hop_length + transform.win_length
    audio, sr = torchaudio.load(filepath)

    # chunk file
    samples = max(clip_length, audio.size(1))
    steps = transform.sample_rate
    startings = range(0, samples - steps, steps)

    # create mini batch from chunks
    x = torch.zeros((len(startings), clip_length))
    for ind, start in enumerate(startings):
        signal = audio[:, start:start + clip_length]
        signal = fix_length(signal, clip_length)
        x[ind] = signal

    # forward pass
    with torch.no_grad():
        x = x.to(device)
        x = transform(x).unsqueeze(1) + 1
        x = x.log()
        x = model(x)
        x = x.mean(dim=0)

    return x


def cosine_based(rep0, rep1):
    rep0 = rep0 / rep0.norm()
    rep1 = rep1 / rep1.norm()
    return -1 * (rep0 - rep1).norm()


def distance_based(rep0, rep1):
    return -1 * (rep0 - rep1).norm()


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def EER_metric(model, transform, num_frames, criterion, device, test_csv):
    print('-' * 20 + f'EER evaluation' + '-' * 20)
    model.eval()

    with open(test_csv) as f:
        eval_data = list(csv.reader(f, delimiter=' '))

    # select similarity measure based on criterion
    if criterion in ['cosface', 'psge2e']:
        sim_scorer = cosine_based
    elif criterion in ['prototypical']:
        sim_scorer = distance_based

    # calculate scores
    labels, scores = [], []
    cache = {}
    for ind, (label, utt0, utt1) in enumerate(tqdm(eval_data)):
        labels.append(int(label))

        if utt0 in cache.keys():
            rep0 = cache[utt0]
        else:
            rep0 = utternace_repr(model, transform, num_frames, device, utt0)
            cache[utt0] = rep0

        if utt1 in cache.keys():
            rep1 = cache[utt1]
        else:
            rep1 = utternace_repr(model, transform, num_frames, device, utt1)
            cache[utt1] = rep1

        scores.append(sim_scorer(rep0, rep1).item())

    eer = compute_eer(labels, scores)
    model.train()
    return eer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training options')
    parser = add_args(parser)
    args = parser.parse_args()
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
        ids.append([i for i in utt0.split('/') if i.startswith('id0')][0])

        rep1 = utternace_repr(
            model,
            feature_extractor,
            args.num_frames,
            device,
            utt1
        )
        reps.append(rep1.cpu().numpy())
        ids.append([i for i in utt1.split('/') if i.startswith('id0')][0])

    id_set = sorted(list(set(ids)))
    id_colors = [np.random.rand(3,) for i in id_set]
    ids_index = [id_set.index(i) for i in ids]

    tsne = TSNE(n_components=2, random_state=0)
    reps_2d = tsne.fit_transform(reps)

    for spk_id, rep in zip(ids_index, reps_2d):
        c = id_colors[spk_id]
        plt.scatter(rep[0], rep[1], c=c)
    plt.savefig('tmp.png', dpi=600)
