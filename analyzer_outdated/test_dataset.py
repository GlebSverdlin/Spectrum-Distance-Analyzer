import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy

training_path = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/training/training_set.csv"
testing_path = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/testing/testing_set.csv"
validating_path = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/validation/validating_set.csv"


class training_dataset(Dataset):
    def __init__(self, data, labels):
        self.features = pd.read_csv(data, usecols=['x0', 'x1', 'x2', 'x3','x4', 'x5'])
        self.features=self.features.to_numpy(dtype='f4')

        self.labels = pd.read_csv(labels)
        self.labels = self.labels.to_numpy(dtype='f4')
    def __len__(self):
        return(len(self.features))

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class testing_dataset(Dataset):
    def __init__(self, data, labels):
        self.features = pd.read_csv(data, usecols=['x0', 'x1', 'x2', 'x3','x4', 'x5'])
        self.features=self.features.to_numpy(dtype='f4')

        self.labels = pd.read_csv(labels)
        self.labels = self.labels.to_numpy(dtype='f4')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class validating_dataset(Dataset):
    def __init__(self, data, labels):
        self.features = pd.read_csv(data, usecols=['x0', 'x1', 'x2', 'x3','x4', 'x5'])
        self.features=self.features.to_numpy(dtype='f4')

        self.labels = pd.read_csv(labels)
        self.labels = self.labels.to_numpy(dtype='f4')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

