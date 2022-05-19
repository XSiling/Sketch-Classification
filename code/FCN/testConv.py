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
from tqdm import tqdm
import torch.optim as optim
from torch.nn import NLLLoss
import random
import datetime
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--valid_interval', type=int, default=130000)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--save_path', type=str, default='model.pth')
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--log_file', type=str, default='log.txt')
parser.add_argument('--loss_file',type=str,default='loss.txt')
args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=10, kernel_size=3, stride=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(10, 20, 3, 1)
        self.max_pool2 = nn.MaxPool1d(3, 1)

        self.fconv1 = nn.Conv1d(20, 40, 1, 1)
        self.fconv2 = nn.Conv1d(40, 25, 1, 1)

    def forward(self, x):
        # pdb.set_trace()
        x = F.tanh(self.conv1(x))
        x = self.max_pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.max_pool2(x)

        x = F.tanh(self.fconv1(x))
        x = F.relu(self.fconv2(x))

        x = torch.sum(x, dim=2)
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


def train(model, opt, train_data, valid_data):
    loss_function = NLLLoss()
    train_length = len(train_data['X'])
    valid_length = len(valid_data['X'])
    # pdb.set_trace()
    best_acc = 0
    for epoch in tqdm(range(args.epoch)):
        # train
        loss_epoch = 0.
        opt.zero_grad()
        for ii in tqdm(range(train_length)):
            input = torch.from_numpy(train_data['X'][ii])
            label = torch.LongTensor(np.zeros((1,)))
            label[0] = train_data['Y'][ii]
            # label[0, train_data['Y'][ii]] = 1
            input = torch.tensor(input, dtype=torch.float32).reshape(1, input.shape[0], input.shape[1])
            input = input.permute(0, 2, 1).to(DEVICE)
            label = torch.tensor(label, dtype=torch.long)
            label = label.to(DEVICE)
            pred = model(input)
            # pdb.set_trace()
            loss = F.cross_entropy(pred, label)
            # loss = F.nll_loss(F.log_softmax(pred), label)
            loss_epoch += loss

            if ii % args.batch_size == 0 and ii !=0:
                loss_epoch /= args.batch_size
                # update parameters
                loss_epoch.backward()
                opt.step()
                print("\nLoss: ", loss_epoch.item())
                writer = str(loss_epoch.item()) + "\n"
                with open(args.loss_file, 'a') as fp:
                    fp.write(writer)
                torch.cuda.empty_cache()
                loss_epoch = 0
                opt.zero_grad()
                
            #print("Epoch: ", epoch, "| Loss: ", loss_epoch)

        # valid
        if True:
            with torch.no_grad():
                print("\nBegin Validation:")
                n_correct = 0
                for jj in tqdm(range(valid_length)):
                    input = torch.from_numpy(valid_data['X'][jj])
                    label = torch.LongTensor(np.zeros((1,)))
                    label[0] = valid_data['Y'][jj]
                    input = torch.tensor(input, dtype=torch.float32).reshape(1, input.shape[0], input.shape[1])
                    input = input.permute(0, 2, 1).to(DEVICE)
                    label = torch.tensor(label, dtype=torch.long)
                    label = label.to(DEVICE)
                    pred = model(input)
                    pred_class = torch.argmax(pred)
                    if pred_class.item() == label[0].item():
                        n_correct += 1
                print("\nValidation Accuracy: ", n_correct/valid_length)
                writer = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S') + "Epoch:" + str(epoch) + "Validation Accuracy:" + str(n_correct/valid_length) + "\n"
                with open(args.log_file, 'a') as fp:
                    fp.write(writer)
                if n_correct/valid_length > best_acc:
                    best_acc = n_correct/valid_length
                    torch.save(model.state_dict(), args.save_path)


def test(model, test_data):
    test_length = len(test_data['X'])
    n_correct = 0
    for ii in range(test_length):
        input = torch.from_numpy(test_data['X'][ii])
        label = torch.LongTensor(np.zeros((1,)))
        label[0] = test_data['Y'][ii]
        input = torch.tensor(input, dtype=torch.float32).reshape(1, input.shape[0], input.shape[1])
        input = input.permute(0, 2, 1).to(DEVICE)
        label = torch.tensor(label, dtype=torch.long)
        label = label.to(DEVICE)
        pred = model(input)
        pred_class = torch.argmax(pred)
        if pred_class.item() == label[0].item():
            n_correct += 1
    print("Test Accuracy: ", n_correct/test_length)


if __name__ == '__main__':
    train_data, valid_data, test_data = load_data(args.data_path)
    randnum = random.randint(0,100)
    # begin training
    model = FCNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    
    random.seed(randnum)
    random.shuffle(train_data['X'])
    random.seed(randnum)
    random.shuffle(train_data['Y'])

    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(valid_data['X'])
    random.seed(randnum)
    random.shuffle(valid_data['Y'])

    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(test_data['X'])
    random.seed(randnum)
    random.shuffle(test_data['Y'])

    train(model, opt, train_data, valid_data)
    test(model, test_data)
