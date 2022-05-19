import numpy as np
import os
import random
import pdb
from tqdm import tqdm
# from quickdraw import QuickDrawData
import json
import pickle
# import svgwrite
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim as optim
from torch.nn import NLLLoss
import random
import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../npydata/')
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--valid_interval', type=int, default=130000)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--save_path', type=str, default='model.pth')
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--log_file', type=str, default='log.txt')
parser.add_argument('--loss_file', type=str, default='loss.txt')
args = parser.parse_args()


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(10, 40, 3, 1)
        self.max_pool2 = nn.MaxPool2d(3, 1)
        self.conv3 = nn.Conv2d(40, 80, 3, 1)
        self.max_pool3 = nn.MaxPool2d(3, 1)

        self.fconv1 = nn.Conv2d(80, 160, 1, 1)
        self.fconv2 = nn.Conv2d(160, 25, 1, 1)

    def forward(self, x):
        #pdb.set_trace()
        x = F.tanh(self.conv1(x))
        x = self.max_pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.max_pool2(x)
        x = F.tanh(self.conv3(x))
        x = self.max_pool3(x)

        x = F.tanh(self.fconv1(x))
        x = F.relu(self.fconv2(x))

        #db.set_trace()
        x = torch.sum(x, dim=(2, 3))
        # pdb.set_trace()
        # x = F.softmax(x)
        return x


def load_data(data_path):
    with open(data_path + 'train_data.pkl', 'rb') as fp:
        train_data = pickle.load(fp, encoding='latin1')
    with open(data_path + 'valid_data.pkl', 'rb') as fp:
        valid_data = pickle.load(fp, encoding='latin1')
    with open(data_path + 'test_data.pkl', 'rb') as fp:
        test_data = pickle.load(fp, encoding='latin1')
    return train_data, valid_data, test_data


def train(model, opt, train_loader, valid_loader):
    loss_function = NLLLoss()
    best_acc = 0.
    for epoch in tqdm(range(args.epoch)):
        opt.zero_grad()
        loss_epoch = 0.
        for _, (input, label) in tqdm(enumerate(train_loader)):
            pred = model(input)
            loss = F.cross_entropy(pred, label)
            loss_epoch += loss
            loss.backward()
            print(loss.item())
            opt.step()
            opt.zero_grad()
            torch.cuda.empty_cache()
        # loss_epoch = loss_epoch/len(train_loader)
        print("Epoch:", epoch, "| Loss:", loss_epoch.item())

        #validation
        with torch.no_grad():
            n_correct = 0
            sum = 0
            for _, (input, label) in enumerate(valid_loader):
                pred = model(input)
                pred_class = pred.argmax(1)
                correct_num = (pred_class == label).sum()
                n_correct += correct_num
                sum += pred_class.shape[0]
            acc = n_correct/sum
            with open('log.txt', 'a') as fp:
                fp.write(datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S') + str(epoch) + "validation acc:" + str(acc))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'best_model'+str(epoch)+'.pth')


        #train for this epoch




if __name__ == '__main__':
    train_data, valid_data, test_data = load_data('../npydata/')
    train_X = np.array(train_data['X'])
    train_Y = np.array(train_data['Y'])
    valid_X = np.array(valid_data['X'])
    valid_Y = np.array(valid_data['Y'])
    test_X = np.array(test_data['X'])
    test_Y = np.array(test_data['Y'])
    train_X = torch.from_numpy(train_X).float().permute(0, 3, 1, 2)
    train_Y = torch.from_numpy(train_Y).long()
    valid_X = torch.from_numpy(valid_X).float().permute(0, 3, 1, 2)
    valid_Y = torch.from_numpy(valid_Y).long()
    test_X = torch.from_numpy(test_X).float().permute(0, 3, 1, 2)
    test_Y = torch.from_numpy(test_Y).long()

    train_set = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    valid_set = TensorDataset(valid_X, valid_Y)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=True)
    test_set = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    model = FCNet()

    opt = optim.Adam(model.parameters(), lr=args.lr)
    train(model, opt, train_loader, valid_loader)