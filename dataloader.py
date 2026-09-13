import os
import pandas as pd
from secret import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import torch
import random

logger = logging.getLogger(__name__)


class SpectralDataset(Dataset):
    def __init__(self, format, purpose):
        match format:
            case "ap":
                self.data_dir = dataset_ap
            case "aspcap":
                self.data_dir = dataset_aspcap
        self.spectra = []
        labels = []
        for i in os.scandir(self.data_dir):
            self.spectra.append(i.name)
            labels.append(pd.read_csv(self.data_dir+i.name, usecols=["planet"], index_col=False))

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
        
        print(np.shape(self.spectra))
        print(self.spectra[0])
        print(type(self.spectra))
        train_length = 1600

        positives = []
        negatives = []
 
        for i, item in enumerate(self.spectra):
            if labels[i].to_numpy()[0] == 1:
                positives.append(self.spectra[i])
            else:
                negatives.append(self.spectra[i])
 
        print(f'Pos: {len(positives)}')
        print(f'Neg: {len(negatives)}')

        j = int(len(self.spectra) / 8)
    
        self.train = negatives[:int(train_length/2)]+positives[:int(train_length/2)]
        self.eval = negatives[int(train_length/2):]+positives[int(train_length/2):]
        random.shuffle(self.train)
                  
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
            flux_resc.append((i - min) * 10 / (max - min))
        return flux_resc

    def __getitem__(self, idx):
        filename = self.spectra[idx]
        path = str(self.data_dir + filename)
        data = pd.read_csv(path, usecols=["flux"], index_col=False)
        label = pd.read_csv(path, usecols=["planet"], index_col=False)
        label = label.head(1)
        data = data.to_numpy(dtype=np.double, na_value=0.0)
        # data = self.rescale_spectrum(data, self.spec_max, self.spec_min)
        # for i in self.zeroes:
        #     data[i] = [0.0]
        data = np.asarray(data).flatten()
        label = label.to_numpy(dtype=np.double)
        label = np.asarray(label[0]).flatten() 
        return data, label
