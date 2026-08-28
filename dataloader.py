import os
import pandas as pd
from secret import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SpectralDataset(Dataset):
    def __init__(self, format, purpose):
        match format:
            case "ap":
                self.data_dir = dataset_ap
            case "aspcap":
                self.data_dir = dataset_aspcap
        self.spectra = []
        for i in os.scandir(self.data_dir):
            self.spectra.append(i.name)

        self.spec_max = 0
        self.spec_min = 10

        self.tables = os.scandir(self.data_dir)
        for self.item in self.tables:
            data = pd.read_csv(self.item, usecols=["flux"], index_col=False)

            local_max = np.nanmax(data)
            local_min = np.nanmin(data)

            if self.spec_max < local_max:
                self.spec_max = local_max
            if self.spec_min > local_min:
                self.spec_min = local_min

        j = int(len(self.spectra) / 8)
        self.eval = self.spectra[:j]
        self.train = self.spectra[j:]
        match purpose:
            case "train":
                self.spectra = self.train
                print(f"purpose:", purpose, ", length:", len(self.spectra))
            case "eval":
                self.spectra = self.eval
                print(f"purpose:", purpose, ", length:", len(self.spectra))

    def __len__(self):
        return len(self.spectra)

    def rescale_spectrum(self, flux, max, min):
        self.zeroes = [i for i in range(0, len(flux)) if flux[i] == 0.0]
        flux_resc = []
        for i in flux:
            flux_resc.append((i - min) / (max - min))
        return flux_resc

    def __getitem__(self, idx):
        filename = self.spectra[idx]
        path = str(self.data_dir + filename)
        data = pd.read_csv(path, usecols=["flux"], index_col=False)
        label = pd.read_csv(path, usecols=["planet"], index_col=False)
        label = label.head(1)
        data = data.to_numpy(dtype=float, na_value=0.0)
        data = self.rescale_spectrum(data, self.spec_max, self.spec_min)
        for i in self.zeroes:
            data[i] = [0.0]
        data = np.asarray(data)
        label = label.to_numpy(dtype=float)
        return data, label
