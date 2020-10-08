import os
import csv
from collections import defaultdict
import random
import json
from glob import glob

import torch
import torch.nn.functional as F
import librosa
import numpy as np
from scipy import signal
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
            feat_type,
            musan_path,
            rir_path,
            augmet_prob
    ):
        self.sample_rate = sample_rate
        self.n_win_length = int(win_length * sample_rate)
        self.n_hop_length = int(hop_length * sample_rate)
        self.n_frames = n_frames
        self.n_fft = n_fft
        self.n_filterbanks = n_filterbanks
        self.feat_type = feat_type
        self.add_musan = musan_path
        self.add_rir = rir_path
        self.augmet_prob = augmet_prob

        self.clip_length = (n_frames-3) * self.n_hop_length + self.n_win_length

        # init augmentation stuff
        if musan_path:
            self.noise_snr_range = {
                'noise': (0, 15),
                'speech': (15, 20),
                'music': (5, 15)
            }
            self.num_noise = {
                'noise': (1, 1),
                'speech': (3, 7),
                'music': (1, 1)
            }
            self.musan_files  = defaultdict(list)
            for filepath in glob(os.path.join(musan_path, '*/*/*.wav')):
                noise_cat = filepath.split('/')[-3]
                self.musan_files[noise_cat].append(filepath)

        if rir_path:
            self.rir_files = glob(os.path.join(rir_path, '*/*/*.wav'))

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

        # augmentation
        if random.random() < self.augmet_prob:
            noise_type = random.randint(0, 3)
            if noise_type == 0:
                rir_file = random.choice(self.rir_files)
                rir, _ = librosa.load(rir_file, sr=self.sample_rate)
                rir /= np.sqrt(np.sum(rir ** 2))
                x = signal.convolve(x, rir, mode='same')
            elif noise_type == 1:
                x = self.add_noise(x, 'music')
            elif noise_type == 2:
                x = self.add_noise(x, 'speech')
            elif noise_type == 3:
                x = self.add_noise(x, 'noise')

        return self.audio_to_feat(x)

    def add_noise(self, x, noise_cat):
        x_db = 10 * np.log10(np.mean(x ** 2) + 1e-4)
        num_noise = self.num_noise[noise_cat]
        file_list = random.sample(
            self.musan_files[noise_cat],
            random.randint(*num_noise)
        )
        for filepath in file_list:
            # load noise
            noise, _ = librosa.load(filepath, sr=self.sample_rate)
            filesize = noise.shape[-1]
            if filesize >= self.clip_length:
                offset = np.random.randint(
                    0,
                    max(0, filesize - self.clip_length)
                )
                noise = noise[offset:offset + self.clip_length]
            else:
                noise = self.fix_length(noise)
            # scale and add to original signal
            noise_snr = random.uniform(*self.noise_snr_range[noise_cat])
            noise_db = 10 * np.log10(np.mean(noise ** 2) + 1e-4)
            noise *= np.sqrt(10 ** ((x_db - noise_db - noise_snr) / 10))
            x += noise
        return x

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
            musan_path,
            rir_path,
            augmet_prob,
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
            feat_type,
            musan_path,
            rir_path,
            augmet_prob
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
    ds = VoxCelebDataset(
        16000,
        .025,
        .01,
        200,
        512,
        40,
        'spect',
        '/data/musan',
        '/data/RIRS_NOISES/simulated_rirs',
        .8,
        'dev',
        '/data/voxceleb2_dev.csv',
        3
    )

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=8, drop_last=True)
    for x, y in tqdm(dl):
        pass
    print(x)
    print(y)
