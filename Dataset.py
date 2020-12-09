import torch
import numpy as np
class Dataset_Action_Obs_Y(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data = None, data_path = None):
        'Initialization'
        if data is not None:
            if len(data) != 3:
                raise Exception("data need to be a list of 2, first is input, second is output")
            else:
                self.action = data[0]
                self.obs = data[1]
                self.y = data[2]
        elif data_path is not None:
            if len(data_path) != 3:
                raise Exception("datapath need to be a list of 2, first is input, second is output")
            else:
                self.action = np.genfromtxt(data_path[0], delimiter=',')
                self.obs = np.genfromtxt(data_path[1], delimiter=',')
                self.y = np.genfromtxt(data_path[2], delimiter=',')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.action)

    def __getitem__(self, index):
        'Generates one sample of data'

        Action = self.action[index]
        Obs = self.obs[index]
        y = self.y[index]
        return [Action, Obs], y


class Dataset_X_Y(torch.utils.data.Dataset):
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

        x = self.x[index]
        y = self.y[index]
        return x, y