import os

import numpy as np


def get_training_data(annotation_path, data_path,load_previous):
    if load_previous == True and os.path.isfile(data_path):
        data = np.load(data_path)
        print('Loading training data from ' + data_path)
        return data['x'], data['adj'], data['y']
    
    x = np.loadtxt(annotation_path['x'])
    adj = np.loadtxt(annotation_path['adj'])
    y = np.loadtxt(annotation_path['y'])
    
    return x,adj,y
    
