# coding: utf-8

import os
import time
import random
import random
import torch
import torchaudio

import librosa
import numpy as np
import soundfile as sf
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 validation=False,
                 ):

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [(path, int(label.replace('p', ''))) for path, label in _data_list]

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.validation = validation
        self.max_mel_length = 192

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_tensor, label, wave_16k_tensor = self._load_data(data)
        spk_emb = spk_model.encode_batch(wave_16k_tensor.unsqueeze(0)).squeeze()
        return mel_tensor, label, spk_emb

    def _load_data(self, path):
        wave_tensor, label, wave_16k_tensor= self._load_tensor(path)

        if not self.validation:  # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor

        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, label, wave_16k_tensor

    def _load_tensor(self, data):
        wave_path, label = data
        label = int(label)
        wave, sr = librosa.load(wave_path, sr=self.sr)
        if sr != self.sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sr)
        wave_16k = librosa.resample(wave, orig_sr=self.sr, target_sr=16000)
        wave_tensor = torch.from_numpy(wave).float()
        wave_16k_tensor = torch.from_numpy(wave_16k).float()
        return wave_tensor, label, wave_16k_tensor


class Collater(object):
    def __init__(self):
        self.max_mel_length = 192
        self.spk_emb_len = 192

    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        labels = torch.zeros((batch_size)).long()
        spk_embs = torch.zeros((batch_size, self.spk_emb_len)).float()


        for bid, (mel, label, spk_emb) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            labels[bid] = label
            spk_embs[bid] = spk_emb

        mels= mels.unsqueeze(1)
        return mels, labels, spk_embs

def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     spkr_model=None
                     ):
    global spk_model
    spk_model = spkr_model
    dataset = MelDataset(path_list, validation=validation)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
