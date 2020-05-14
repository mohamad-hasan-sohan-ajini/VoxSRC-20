import os
import csv
from collections import defaultdict
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tinytag import TinyTag


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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, speaker_index):
        raise NotImplementedError('__getitem__ not implemented')

    def fix_length(self, signal):
        lack = self.clip_length - signal.size(1)
        pad_left = pad_right = lack // 2
        pad_left += 1 if lack % 2 else 0
        return F.pad(signal, (pad_left, pad_right))

    def load_audio(self, filepath):
        # get duration
        tag = TinyTag.get(filepath)
        filesize = int(tag.duration * tag.samplerate)
        if filesize >= filesize:
            # random crop
            samples = min(self.clip_length, filesize)
            starting = torch.randint(max(1, filesize - samples), (1, ))
            x, sr = torchaudio.load(
                filepath,
                offset=starting,
                num_frames=self.clip_length
            )
        else:
            # load whole audio
            x, sr = torchaudio.load(filepath)
            # fix length
            x = self.fix_length(x)
        return x


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
        filename = random.choice(self.data[speaker])
        audio_segment = self.load_audio(filename)
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
        filenames = random.choices(self.data[speaker], k=self.spk_samples)
        audio_segments = [self.load_audio(i) for i in filenames]
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
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
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
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
    for x, y in tqdm(dl):
        pass
    print(x)
    print(y)
