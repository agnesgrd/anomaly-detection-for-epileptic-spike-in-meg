import torch
from torch.utils.data import Dataset

import sys 
import numpy as np

from utils import z_score_normalize, load_obj


class TimeSeriesDataSet(Dataset):
    """This is a custom dataset class."""

    def __init__(self, X_ids, dim, path, out="XX", aug=False):
        self.X = X_ids
        self.path = path
        self.out = out
        self.dim = dim
        self.aug = aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
        win = np.array(self.X[index])[0]
        sub = np.array(self.X[index])[1]

        prefixe = "data_raw_"
        suffixe = "_b3_windows_bi"
        path_data = self.path

        # Opens windows_bi binary file
        f = open(path_data + prefixe + str("{:03d}".format(sub)) + suffixe)
        # Set cursor position to 30 (nb time points)*274 (nb channels)*windows_id*4 because data is stored as float64 and dtype.itemsize = 8
        f.seek(self.dim[0] * self.dim[1] * win * 4)
        # From cursor location, get data from 1 window
        sample = np.fromfile(f, dtype="float32", count=self.dim[0] * self.dim[1])
        # Reshape to create a 2D array (data from the binary file is just a vector)
        sample = sample.reshape(self.dim[1], self.dim[0])
        
        if sys.argv[2] == "wavelets":
            sample = np.swapaxes(sample,0,1)

        sample = z_score_normalize(sample)

        # print(np.mean(sample))
        # sample = normalize_by_window(sample)
        # sample = stats.zscore(sample,axis=None)

        lab_t = load_obj("data_raw_" + str("{:03d}".format(sub)) + "_b3_labels_t.pkl", self.path)

        # Add third dimension to be able to feed to CNN (CNN need 3 dim, image x, image y, RGB channels), here last dimension=1
        if len(self.dim) == 3:
            sample = np.expand_dims(sample, axis=0)
            # sample = np.expand_dims(sample,axis=-1)

        _x = sample
        _y = lab_t[win] #np.array(self.X[index])[2]

        if self.aug:
            noise = np.random.normal(loc=0.0, scale=0.2, size=_x.shape)
            _x = _x + noise

        # print("Valeurs X",_x)
        if self.out == "XX":
            return torch.tensor(_x, dtype=torch.float32), torch.tensor(_x, dtype=torch.float32)
        elif self.out == "Xy":
            return torch.tensor(_x, dtype=torch.float32), torch.tensor(_y, dtype=torch.float32)
