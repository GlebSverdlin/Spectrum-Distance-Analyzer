import os
import pandas as pd
from secret import *
from torch.utils.data import Dataset, DataLoader

class SpectralDataset(Dataset):
    def __init__(self, datatype):
        match datatype:
            case 'ap': self.data_dir = dataset_ap
            case 'aspcap': self.data_dir = dataset_aspcap
        self.spectra = []
        for i in os.scandir(self.data_dir): self.spectra.append(i.name)

    def __len__(self):
        return len(self.spectra)
        
    def __getitem__(self, idx):
        filename = self.spectra[idx]
        path = str(self.data_dir+filename)
        data = pd.read_csv(path, usecols=['flux'], index_col=False)
        label = pd.read_csv(path, usecols=['planet'], index_col=False)
        label = label.head(1)
        data = data.to_numpy(dtype=float, na_value = 0.0)
        label = label.to_numpy(dtype=float)
        return data, label

