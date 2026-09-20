import os
import time
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
        self.dataset_list = []
        labels = []
       
        self.tables = os.scandir(self.data_dir)

        for item in self.tables:
            self.spectra.append(item.name)
            item_path = os.path.join(self.data_dir, item.name)

            item_data_col = pd.read_csv(item_path, usecols=["flux"], index_col=False)
            item_label_col = pd.read_csv(item_path, usecols=["planet"], index_col=False)

            item_label = item_label_col.head(1)
            item_data = item_data_col.to_numpy(dtype=np.double, na_value=0.0)

            labels.append(item_label)#pd.read_csv(self.data_dir+i.name, usecols=["planet"], index_col=False))
            
            data = np.asarray(item_data).flatten()
            label = item_label.to_numpy(dtype=np.double)

            label = np.asarray(label[0]).flatten()      
            
            self.dataset_list.append({'data':data, 'label':label})

        # self.spec_max = 0
        # self.spec_min = 10
        #
        # for self.item in self.tables:
        #     data = pd.read_csv(self.item, usecols=["flux"], index_col=False)
        #
        #     local_max = np.nanmax(data)
        #     local_min = np.nanmin(data)
        #
        #     if self.spec_max < local_max:
        #         self.spec_max = local_max
        #     if self.spec_min > local_min:
        #         self.spec_min = local_min
        #
        train_length = 1600

        positives = []
        negatives = []
 
        for i, item in enumerate(self.dataset_list):
            if labels[i].to_numpy()[0] == 1:
                positives.append(self.dataset_list[i])
            else:
                negatives.append(self.dataset_list[i])
 
        print(f'Pos: {len(positives)}')
        print(f'Neg: {len(negatives)}')

        # j = int(len(self.spectra) / 8)
        #
        self.train_plain = negatives[:int(train_length/2)] + positives[:int(train_length/2)]
        self.eval = negatives[int(train_length/2):] + positives[int(train_length/2):]
        # random.shuffle(self.train)
        self.train = []
        
        for iter in range(int(len(self.train_plain)/2)):
            self.train.append(self.train_plain[iter])
            self.train.append(self.train_plain[len(self.train_plain)-1-iter])

        match purpose:
            case "train":
                self.spectra = self.train
                print(f"purpose:", purpose, ", length:", len(self.spectra))
            case "eval":
                self.spectra = self.eval
                print(f"purpose:", purpose, ", length:", len(self.spectra))
            case _:
                raise ValueError("Wrong purpose train/eval")

    def __len__(self):
        # return len(self.spectra)
        return len(self.spectra)

    # def rescale_spectrum(self, flux, max, min):
    #     self.zeroes = [i for i in range(0, len(flux)) if flux[i] == 0.0]
    #     flux_resc = []
    #     for i in flux:
    #         flux_resc.append((i - min) * 10 / (max - min))
    #     return flux_resc
    #
    def __getitem__(self, idx):
        time_start = time.perf_counter()
        # filename = self.spectra[idx]
        # path = str(self.data_dir + filename)
        # data = pd.read_csv(path, usecols=["flux"], index_col=False)
        # label = pd.read_csv(path, usecols=["planet"], index_col=False)
        # label = label.head(1)
        # data = data.to_numpy(dtype=np.double, na_value=0.0)
        # data = self.rescale_spectrum(data, self.spec_max, self.spec_min)
        # for i in self.zeroes:
        #     data[i] = [0.0]
        # data = np.asarray(data).flatten()
        # label = label.to_numpy(dtype=np.double)
        # label = np.asarray(label[0]).flatten()
        data = self.spectra[idx]['data']
        label = self.spectra[idx]['label']
        time_end = time.perf_counter()
        # print (f"Get item time {time_end - time_start}")
        return data, label
