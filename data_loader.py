import os
import csv
from collections import defaultdict
import random
import json

import torch
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tinytag import TinyTag


class FeatureExtractor(object):
    def __init__(
            self,
            sample_rate,
            win_length,
            hop_length,
            n_frames,
            n_fft,
            n_filterbanks,
            feat_type
    ):
        self.sample_rate = sample_rate
        self.n_win_length = int(win_length * sample_rate)
        self.n_hop_length = int(hop_length * sample_rate)
        self.n_frames = n_frames
        self.n_fft = n_fft
        self.n_filterbanks = n_filterbanks
        self.feat_type = feat_type

        self.clip_length = (n_frames-3) * self.n_hop_length + self.n_win_length

    def load_audio_4train(self, filepath, filesize):
        if filesize >= self.clip_length:
            # random crop
            offset = np.random.randint(0, max(0, filesize - self.clip_length))
            x, _ = librosa.load(
                filepath,
                offset=offset / self.sample_rate,
                duration=self.clip_length / self.sample_rate,
                sr=None
            )
        else:
            # load whole audio and fix length
            x, _ = librosa.load(filepath, sr=None)
            x = self.fix_length(x)
        return self.audio_to_feat(x)

    def load_audio_4test(self, filepath):
        x, _ = librosa.load(filepath, sr=None)
        return self.audio_to_feat(x)

    def audio_to_feat(self, x):
        # feature extraction
        if self.feat_type == 'mel':
            feature = librosa.feature.melspectrogram(
                y=x,
                n_fft=self.n_fft,
                hop_length=self.n_hop_length,
                win_length=self.n_win_length,
                n_mels=self.n_filterbanks
            )
        elif self.feat_type == 'spect':
            D = librosa.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.n_hop_length,
                win_length=self.n_win_length,
            )
            feature, _ = librosa.magphase(D)

        # normalize by log
        feature = np.log1p(feature)

        # CMVN normalization
        feature = torch.FloatTensor(feature)
        with torch.no_grad():
            mean = feature.mean()
            std = feature.std()
            feature.add_(-mean)
            feature.div_(std)

        return feature

    def fix_length(self, signal):
        lack = self.clip_length - signal.shape[0]
        pad_left = pad_right = lack // 2
        pad_left += 1 if lack % 2 else 0
        return np.concatenate(
            [np.zeros(pad_left), signal, np.zeros(pad_right)]
        )


class VoxCelebDataset(Dataset):
    def __init__(
            self,
            sample_rate,
            win_length,
            hop_length,
            n_frames,
            n_fft,
            n_filterbanks,
            feat_type,
            mode,
            csv_path,
            samples_per_speaker
    ):
        self.feature_extractor = FeatureExtractor(
            sample_rate,
            win_length,
            hop_length,
            n_frames,
            n_fft,
            n_filterbanks,
            feat_type
        )
        self.mode = mode

        if mode == 'dev':
            self.developing_mode_init(csv_path, samples_per_speaker)
        elif mode == 'eval':
            self.evaluation_mode_init(csv_path)
        else:
            raise ValueError('dataset mode must be "dev" or "eval".')

    def developing_mode_init(self, csv_path, samples_per_speaker):
        self.samples_per_speaker = samples_per_speaker

        self.data = defaultdict(list)
        with open(csv_path) as f:
            for speaker_id, filepath in csv.reader(f, delimiter=' '):
                self.data[speaker_id].append(filepath)
        speaker_list = sorted(list(self.data.keys()))
        self.index2speaker = {ind: spk for ind, spk in enumerate(speaker_list)}
        self.speaker2index = {spk: ind for ind, spk in enumerate(speaker_list)}

        # read file sizes
        if os.path.exists('filesize.json'):
            with open('filesize.json') as f:
                self.filesize = json.load(f)
        else:
            print('read file tags...')
            self.filesize = {}
            for file_list in tqdm(self.data.values()):
                for filepath in file_list:
                    tag = TinyTag.get(filepath)
                    filesize = int(tag.duration * tag.samplerate)
                    self.filesize[filepath] = filesize
            with open('filesize.json', 'w') as f:
                json.dump(self.filesize, f)

    def evaluation_mode_init(self, csv_path):
        self.data = []
        print(csv_path)
        with open(csv_path) as f:
            for label, filepath0, filepath1 in csv.reader(f, delimiter=' '):
                self.data.append((label, filepath0, filepath1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'dev':
            return self.dev_getitem(idx)
        else:
            return self.test_getitem(idx)

    def dev_getitem(self, speaker_index):
        speaker = self.index2speaker[speaker_index]
        filenames = random.choices(
            self.data[speaker],
            k=self.samples_per_speaker
        )
        audio_segments = [
            self.feature_extractor.load_audio_4train(i, self.filesize[i])
            for i in filenames
        ]
        label = torch.LongTensor([speaker_index] * self.samples_per_speaker)
        return torch.stack(audio_segments), label

    def test_getitem(self, line_ind):
        label, filepath0, filepath1 = self.data[line_ind]
        return label, filepath0, filepath1


if __name__ == '__main__':
    VoxCelebDataset()
    ds = VoxCelebDataset(
        16000,
        .025,
        .01,
        200,
        512,
        40,
        'mel'
        'train',
        'data/train.csv',
        3
    )

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    for x, y in tqdm(dl):
        pass
    print(x)
    print(y)
