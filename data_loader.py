import os
import csv
from collections import defaultdict
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class VoxCelebDataset(Dataset):
    def __init__(self, csv_path, win_length, hop_length, num_frames):
        self.clip_length = (num_frames-3) * hop_length + win_length

        self.data = defaultdict(list)
        with open(csv_path) as f:
            for speaker_id, filepath in csv.reader(f, delimiter=' '):
                self.data[speaker_id].append(filepath)
        speaker_list = sorted(list(self.data.keys()))
        self.index2speaker = {ind: spk for ind, spk in enumerate(speaker_list)}
        self.speaker2index = {spk: ind for ind, spk in enumerate(speaker_list)}

        # shared buffers
        self.utt_buf = defaultdict(lambda: torch.zeros((1, 0)))
        self.file_ind = defaultdict(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, speaker_index):
        raise NotImplementedError('__getitem__ not implemented')

    def withdraw_buffer(self, spk):
        sample = self.utt_buf[spk][:, :self.clip_length]
        self.utt_buf[spk] = self.utt_buf[spk][:, self.clip_length:]
        return sample

    def fill_up_buffer(self, spk):
        while self.utt_buf[spk].size(1) < self.clip_length:
            # select file from speaker collection and update spk file index
            spk_file_ind = self.file_ind[spk]
            self.file_ind[spk] += 1
            if self.file_ind[spk] == len(self.data[spk]):
                self.file_ind[spk] = 0
                random.shuffle(self.data[spk])
            # load file
            filename = self.data[spk][spk_file_ind]
            x, sr = torchaudio.load(filename)
            # extend the buffer
            self.utt_buf[spk] = torch.cat([self.utt_buf[spk], x], dim=1)


class ClassificationVCDS(VoxCelebDataset):
    def __init__(self, csv_path, win_length, hop_length, num_frames):
        super(ClassificationVCDS, self).__init__(
            csv_path,
            win_length,
            hop_length,
            num_frames
        )

    def __getitem__(self, speaker_index):
        speaker = self.index2speaker[speaker_index]
        self.fill_up_buffer(speaker)
        audio_segment = self.withdraw_buffer(speaker)
        label = torch.LongTensor([speaker_index])
        return audio_segment, label


class MetricLearningVCDS(VoxCelebDataset):
    def __init__(
        self,
        csv_path,
        win_length,
        hop_length,
        num_frames,
        spk_samples=2
    ):
        super(MetricLearningVCDS, self).__init__(
            csv_path,
            win_length,
            hop_length,
            num_frames
        )
        self.spk_samples = spk_samples

    def __getitem__(self, speaker_index):
        speaker = self.index2speaker[speaker_index]
        audio_segments = []
        for i in range(self.spk_samples):
            self.fill_up_buffer(speaker)
            audio_segments.append(self.withdraw_buffer(speaker))
        label = torch.LongTensor([speaker_index])
        return audio_segments, label


def transform(**kwargs):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=kwargs['sample_rate'],
        n_fft=kwargs['win_length'],
        win_length=kwargs['win_length'],
        hop_length=kwargs['hop_length'],
        f_min=0,
        f_max=kwargs['sample_rate'] / 2,
        pad=0,
        n_mels=kwargs['num_filterbanks']
    )


if __name__ == '__main__':
    ds = ClassificationVCDS(
        '/media/aj/wav_train_list.txt',
        400,
        160,
        200
    )
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
    for x, y in tqdm(dl):
        pass
    print(x)
    print(y)

    ds = MetricLearningVCDS(
        '/media/aj/wav_train_list.txt',
        400,
        160,
        200,
        5
    )
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
    for x, y in tqdm(dl):
        pass
    print(x)
    print(y)
