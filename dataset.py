import torch
import numpy as np
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data = None, data_path = None):
        'Initialization'
        if data is not None:
            if len(data) != 2:
                raise Exception("data need to be a list of 2, first is input, second is output")
            else:
                self.x = data[0]
                self.y = data[1]
        elif data_path is not None:
            if len(data_path) != 2:
                raise Exception("datapath need to be a list of 2, first is input, second is output")
            else:
                self.x = np.genfromtxt(data_path[0], delimiter=',')
                self.y = np.genfromtxt(data_path[1], delimiter=',')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'

        X = self.x[index]
        y = self.y[index]
        return X, y