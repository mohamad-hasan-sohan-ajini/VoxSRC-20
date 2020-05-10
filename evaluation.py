import csv

import torch
import torch.nn as nn
import torchaudio
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


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
        x = transform(x).unsqueeze(1)
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


def EER_metric(model, transform, num_frames, criterion, device, eval_csv):
    model.eval()

    with open(eval_csv) as f:
        eval_data = list(csv.reader(f, delimiter=' '))

    # select similarity measure based on criterion
    if criterion in ['cosface', 'psge2e']:
        sim_scorer = cosine_based
    elif criterion in ['prototypical']:
        sim_scorer = distance_based

    # calculate scores
    labels, scores = [], []
    for ind, (label, utt0, utt1) in enumerate(eval_data):
        labels.append(int(label))

        rep0 = utternace_repr(model, transform, num_frames, device, utt0)
        rep1 = utternace_repr(model, transform, num_frames, device, utt1)
        scores.append(sim_scorer(rep0, rep1).item())

    eer = compute_eer(labels, scores)
    model.train()
    return eer
