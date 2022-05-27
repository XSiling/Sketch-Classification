from tqdm import tqdm
import os
import urllib.request
import numpy as np
# fold='dataset/'
import pickle
def load_data():
    for dir in tqdm(os.listdir('dataset/')):
        url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

        path = url + dir.split('_')[1].split('.')[0] + '.npy'
        print(path)
        urllib.request.urlretrieve(path, 'npydata/'+ dir.split('_')[1].split('.')[0] +'.npy')
    # print()
def generate_name():
    dic={}
    i=0
    for dir in tqdm(os.listdir('dataset/')):
        dic[dir.split('_')[1].split('.')[0]]=i
        i+=1
    return dic
def split_data():
    dic=generate_name()
    fold='npydata/'
    train_fig,valid_fig,test_fig=[],[],[]
    train_label,valid_label,test_label=[],[],[]
    for dir in tqdm(os.listdir(fold)):
        data=np.load(fold+dir)
        for i in range(70000):
            train_fig.append(data[i].reshape(28,28,1))
            train_label.append(dic[dir.split('.')[0]])
        for i in range(2500):
            valid_fig.append(data[i+70000].reshape(28,28,1))
            valid_label.append(dic[dir.split('.')[0]])
        for i in range(2500):
            test_fig.append(data[i+72500].reshape(28,28,1))
            test_label.append(dic[dir.split('.')[0]])
    train_data = {'X':train_fig, 'Y':train_label}
    valid_data = {'X':valid_fig, 'Y':valid_label}
    test_data = {'X':test_fig, 'Y':test_label}
    with open('train_data' + '.pkl', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
    with open('valid_data' + '.pkl', 'wb') as f:
        pickle.dump(valid_data, f, pickle.HIGHEST_PROTOCOL)
    with open('test_data' + '.pkl', 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    split_data()