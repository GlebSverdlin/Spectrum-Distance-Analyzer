import os
import pandas as pd
from secret import *

class SpectralDataset(Dataset):
    def __init__(self, datatype):
        match datatype:
            case 'ap': data_dir = dataset_ap
            case 'aspcap': data_dir = dataset_aspcap
        self.spectra = []
        for i in os.scandir(data_dir): spectra.append(i.name)

    def __len__(self):
        return len(spectra)
        
    def __getitem__(self, idx):
        filename = spectra[idx]
        path = str(data_dir+filename)
        data = pd.read_csv(path, usecols='flux')
        label = pd.read_csv(path, usecols='planet')
        return data, label

