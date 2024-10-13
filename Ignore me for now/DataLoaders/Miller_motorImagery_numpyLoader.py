import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import sys
sys.path.append('..')
from Utils.millerUtils import getFeatureTarget, get_all, Dataset_alldat
from sklearn.model_selection import train_test_split  




def loader(root = '../Dataset', stim_id_1 = 11, stim_id_2 = 12, timepoints_length = 3000, channels = np.arange(46), flatten = False, shuffle = True, split = 2, test_size = 0.2):

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
        return X_train, X_test, Y_train, Y_test

    return X, X, Y, Y