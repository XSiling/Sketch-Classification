import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

EPOCH = 100
BATCH_SIZE = 100
LR = 0.001
DOWNLOAD_MNIST = False

BEST_VALID_ACC=0.0

class mydataset(Dataset):
    def __init__(self,filename='ML-sketch/train_data.pkl',transform=torchvision.transforms.ToTensor()) :
        self.dataset=pd.read_pickle(filename)
        self.transform = transform
    def __len__(self):
        return len(self.dataset['X'])
    def __getitem__(self, index):
        fig=self.dataset['X'][int(index)]
        fig=self.transform(fig)
        return fig,(self.dataset['Y'][int(index)])
    def data(self):
        return self.transform(self.dataset['X'])
    def target(self):
        return self.dataset['Y']

train_data=mydataset(filename='npydata/train_data.pkl')
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_data=mydataset(filename='npydata/valid_data.pkl')
test_data=mydataset(filename='npydata/test_data.pkl')
valid_loader=Data.DataLoader(dataset=valid_data,batch_size=BATCH_SIZE, shuffle=True)
test_loader=Data.DataLoader(dataset=test_data,batch_size=BATCH_SIZE, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),)
        self.out = nn.Linear(32 * 7 * 7, 25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output,x

cnn = CNN()

cnn.cuda()      # Moves all model parameters and buffers to the GPU.

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

for valid_x,valid_y in valid_loader:
    print(valid_x.shape)
    print(valid_y.size(0))
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x.cuda().type(torch.cuda.FloatTensor)    # Tensor on GPU
        b_y = y.cuda()    # Tensor on GPU
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy=0.0
    sum_data=0.0
    for valid_x,valid_y in valid_loader:
        valid_x=valid_x.cuda().type(torch.cuda.FloatTensor)
        valid_y=valid_y.cuda() 
        valid_output, last_layer = cnn(valid_x)
        pred_y = torch.max(valid_output, 1)[1].cuda().data  # move the computation in GPU

        accuracy += torch.sum(pred_y == valid_y).type(torch.FloatTensor) 
        sum_data+=valid_y.size(0)
    accuracy=accuracy/sum_data
    if accuracy>BEST_VALID_ACC:
        BEST_VALID_ACC=accuracy
    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| valid accuracy: %.4f' % accuracy)

print('Best valid accuracy: %.4f' %BEST_VALID_ACC)

