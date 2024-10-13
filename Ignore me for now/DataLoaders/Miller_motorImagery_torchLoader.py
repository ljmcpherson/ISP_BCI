import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import sys
sys.path.append('..')
from Utils.millerUtils import getFeatureTarget, get_all, Dataset_alldat
from sklearn.model_selection import train_test_split  
import torch
from torch.utils.data import Dataset, DataLoader  
import torchvision.transforms as transforms  




class CustomDataset(Dataset):  
    def __init__(self, X, Y, transform = None):  
        self.X = X  
        self.Y = Y 
        self.transform = transform 

    def __len__(self):  
        return len(self.Y)  

    def __getitem__(self, idx):  
        x_sample = self.X[idx]    
        y_sample = self.Y[idx]

        if self.transform:  
            x_sample = self.transform(x_sample) 

        return x_sample, y_sample 

def loader(root = '../Dataset', stim_id_1 = 11, stim_id_2 = 12, timepoints_length = 3000, channels = np.arange(46), flatten = False, shuffle = True, split = 2, test_size = 0.2, batch_size = 32, transform = None):

    Dataset_numpy = Dataset_alldat(root)

    real, imagery = get_all(alldat=Dataset_numpy,
                         stim_id_1 = stim_id_1,
                           stim_id_2 = stim_id_2,
                             timepoints_length = timepoints_length,
                               channels = channels)
    

    X,Y = getFeatureTarget(real,
                            imagery,
                              channels = channels,
                                flatten = flatten,
                                  shuffle = shuffle,
                                    split = split)
    
    if test_size is not None:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=2024)
    else:
        X_train = X
        X_test = X
        Y_train = Y
        Y_test = Y

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    X_train = torch.tensor(X_train, dtype=torch.float64, device = device)  
    X_test = torch.tensor(X_test, dtype=torch.float64, device = device)  
    Y_train = torch.tensor(Y_train, dtype=torch.long, device = device)  
    Y_test = torch.tensor(Y_test, dtype=torch.long, device = device)


    train_dataset = CustomDataset(X_train, Y_train, transform = None)  
    test_dataset = CustomDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)


    return train_loader, test_loader