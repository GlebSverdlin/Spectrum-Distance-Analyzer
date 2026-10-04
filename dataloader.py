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

# zeroes = [[8335,8574],[6080,6342],[3274,3584],[0,245]]
zeroes = []
class SpectralDataset(Dataset):
    def __init__(self, format, purpose):
        match format:
            case "ap":
                self.data_dir = dataset_ap
            case "aspcap":
                self.data_dir = dataset_aspcap
            case "aspcap_nz":
                self.data_dir = dataset_aspcap_nz
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
            item_data = item_data_col.head(7514).to_numpy(dtype=np.double, na_value=0.0)

            labels.append(item_label)
            
            data = np.asarray(item_data).flatten()
            label = item_label.to_numpy(dtype=np.double)

            label = np.asarray(label[0]).flatten()

            self.dataset_list.append({'data':data, 'label':label})


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
