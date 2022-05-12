import numpy as np
import os
import util as util
import random
import pdb
from tqdm import tqdm
# from quickdraw import QuickDrawData
import json
import pickle
import svgwrite

data_path = 'dataset/'

def load_data():
    label_reference = {}
    train_set, valid_set, test_set = [], [], []
    train_labels, valid_labels, test_labels = [], [], []
    animal_class = 0
    current_class = 0
    for dir in tqdm(os.listdir('dataset/')):
        print(dir)
        animal_name = dir[:-4].split('_')[1]
        if animal_name in label_reference.keys():
            current_class = label_reference[animal_name]
        else:
            current_class = animal_class
            animal_class += 1
            label_reference[animal_name] = current_class
        data = np.load(data_path + dir, encoding='latin1', allow_pickle=True)
        train_data = data['train']
        valid_data = data['valid']
        test_data = data['test']
        train_set = train_set + train_data.tolist()
        train_labels = train_labels + [current_class for x in range(len(train_data.tolist()))]
        valid_set = valid_set + valid_data.tolist()
        valid_labels = valid_labels + [current_class for x in range(len(valid_data.tolist()))]
        test_set = test_set + test_data.tolist()
        test_labels = test_labels + [current_class for x in range(len(test_data.tolist()))]
    return train_set, train_labels, valid_set, valid_labels, test_set, test_labels, label_reference
  
if __name__ == '__main__':
    train_set, train_labels, valid_set, valid_labels, test_set, test_labels, label_reference = load_data()
    train_data = {'X':train_set, 'Y':train_labels}
    valid_data = {'X':valid_set, 'Y':valid_labels}
    test_data = {'X':test_set, 'Y':test_labels}
    with open('train_data' + '.pkl', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
    with open('valid_data' + '.pkl', 'wb') as f:
        pickle.dump(valid_data, f, pickle.HIGHEST_PROTOCOL)
    with open('test_data' + '.pkl', 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
    with open('class_reference' + '.pkl', 'wb') as f:
        pickle.dump(label_reference, f, pickle.HIGHEST_PROTOCOL)
