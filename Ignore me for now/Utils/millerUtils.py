import numpy as np
import os

# Make sure data / alldata are numpy arrays

def Dataset_alldat(root = '../Dataset'):
    fname = os.path.join(root, 'motor_imagery.npz')
    alldat = np.load(fname, allow_pickle=True)['dat']
    return alldat


def split_classes(data, stim_id_1 = 11, stim_id_2 = 12, timepoints_length = 3000, channels = np.arange(46)):
    
    tongue_data = {'V': [], 't_on': [], 't_off': [], 'stim_id': []}
    hand_data = {'V': [], 't_on': [], 't_off': [], 'stim_id': []}
   
    for i, stim_id in enumerate(data['stim_id']):

        if timepoints_length < data['t_off'][i] - data['t_on'][i]:
           sub = data['t_off'][i] - data['t_on'][i] - timepoints_length
        else:
           sub = 0
        

        t_on = data['t_on'][i]
        t_off = data['t_off'][i] - sub
        if stim_id == stim_id_1:
            tongue_data['V'].append(data['V'][t_on:t_off][:,channels])
            tongue_data['t_on'].append(t_on)
            tongue_data['t_off'].append(t_off)
            tongue_data['stim_id'].append(stim_id)
        elif stim_id == stim_id_2:
            hand_data['V'].append(data['V'][t_on:t_off][:,channels])
            hand_data['t_on'].append(t_on)
            hand_data['t_off'].append(t_off)
            hand_data['stim_id'].append(stim_id)

    # Convert lists to numpy arrays
    tongue_data['V'] = np.array(tongue_data['V'])
    tongue_data['t_on'] = np.array(tongue_data['t_on'])
    tongue_data['t_off'] = np.array(tongue_data['t_off'])
    tongue_data['stim_id'] = np.array(tongue_data['stim_id'])

    hand_data['V'] = np.array(hand_data['V'])
    hand_data['t_on'] = np.array(hand_data['t_on'])
    hand_data['t_off'] = np.array(hand_data['t_off'])
    hand_data['stim_id'] = np.array(hand_data['stim_id'])

    tongue_data['stim_id'] = 1 - (tongue_data['stim_id']==11).astype(int)
    hand_data['stim_id'] = (hand_data['stim_id']==12).astype(int)

    return tongue_data, hand_data

def get_all(alldat, stim_id_1 = 11, stim_id_2 = 12, timepoints_length = 3000, channels = np.arange(46)):

  real = {"tongue" : [], "hand" : []}
  imagery = {"tongue" : [], "hand" : []}
  # Iterate over the datasets
  for i in range(7):
      # Split classes for the current dataset
      real_classes_0, real_classes_1 = split_classes(alldat[i][0], stim_id_1, stim_id_2, timepoints_length = 3000, channels = channels)
      imagery_classes_0,  imagery_classes_1 = split_classes(alldat[i][1], stim_id_1, stim_id_2, timepoints_length = 3000, channels = channels)

      # Append the results to the lists
      real["tongue"].append(real_classes_0)
      imagery["tongue"].append(imagery_classes_0)
      real["hand"].append(real_classes_1)
      imagery["hand"].append(imagery_classes_1)


  real["tongue"] = np.array(real["tongue"])
  imagery["tongue"] = np.array(imagery["tongue"])
  real["hand"] = np.array(real["hand"])
  imagery["hand"] = np.array(imagery["hand"])

  return real, imagery

def getFeatureTarget(real, imagery, channels = np.arange(46), flatten = True, shuffle = True, split = '2'):

    #   split 4 -> imagery hand & real hand & imageryb tongue & real tongue
    #   split 2 -> imagery/real hand & imagery/real tongue (i.e hand & tongue)
    #   split r -> real tongue & real hand
    #   split i -> imagery tongue & imagery hand
 
    X = []
    Y = []

    for i in range(real['tongue'].shape[0]):
        if split == '4':
          X.append(real['tongue'][i]['V'])
          X.append(imagery['tongue'][i]['V'])
          Y.append(np.zeros(real['tongue'][i]['V'].shape[0]))
          Y.append(np.zeros(imagery['tongue'][i]['V'].shape[0]) + 1)
        elif split == '2':
          X.append(real['tongue'][i]['V'])
          X.append(imagery['tongue'][i]['V'])
          Y.append(np.zeros(real['tongue'][i]['V'].shape[0]))
          Y.append(np.zeros(imagery['tongue'][i]['V'].shape[0]))
        elif split == 'r':
          X.append(real['tongue'][i]['V'])
          Y.append(np.zeros(real['tongue'][i]['V'].shape[0]))
        elif split == 'i':
          X.append(imagery['tongue'][i]['V'])
          Y.append(np.zeros(imagery['tongue'][i]['V'].shape[0]))

    for i in range(real['hand'].shape[0]):
        if split == '4':
          X.append(real['hand'][i]['V'])
          X.append(imagery['hand'][i]['V'])
          Y.append(np.ones(real['hand'][i]['V'].shape[0]) + 1)
          Y.append(np.ones(imagery['hand'][i]['V'].shape[0]) + 2)
        elif split == '2':
          X.append(real['hand'][i]['V'])
          X.append(imagery['hand'][i]['V'])
          Y.append(np.ones(real['hand'][i]['V'].shape[0]))
          Y.append(np.ones(imagery['hand'][i]['V'].shape[0]))
        elif split == 'r':
          X.append(real['hand'][i]['V'])
          Y.append(np.ones(real['hand'][i]['V'].shape[0]))
        elif split == 'i':
          X.append(imagery['hand'][i]['V'])
          Y.append(np.ones(imagery['hand'][i]['V'].shape[0]))

    X = np.array(X)
    X = X.reshape(-1, real['tongue'][i]['V'].shape[1], channels.shape[0]).transpose(0, 2, 1)

    Y = np.array(Y)
    Y = Y.reshape(-1)


    if flatten:
      X = X.reshape(-1, X.shape[1]*X.shape[2])

    if shuffle:
      permutation = np.random.permutation(X.shape[0])
      X = X[permutation]
      Y = Y[permutation]

    return X,Y