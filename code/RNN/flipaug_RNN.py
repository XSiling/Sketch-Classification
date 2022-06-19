import torch
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.utils.data as Data
import pandas as pd
import torchvision
# Hyper Parameters
EPOCH = 100               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 100
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.001               # learning rate
DOWNLOAD_MNIST = False   # set to True if haven't download the data

BEST_VALID_ACC=0.0

def generate_flip(X, Y,por=0.3):
    b = torchvision.transforms.functional.hflip(X)
    index = []
    for i in range(b.shape[0]):
        if random.random()<por:
            index.append(i)
    index = torch.from_numpy(np.array(index))
    b1 = b.index_select(0, index)
    c1 = Y.index_select(0, index)
    return b1, c1

class mydataset(Dataset):
    def __init__(self,filename='ML-sketch/train_data.pkl',transform=torchvision.transforms.ToTensor()) :
        self.dataset=pd.read_pickle(filename)
        
        train_X_1, train_Y_1 = generate_flip(self.dataset['X'],self.dataset['Y'],0.3)
        
        self.dataset['X'] = torch.cat((self.dataset['X'], train_X_1), dim=0)
        self.dataset['Y'] = torch.cat((self.dataset['Y'], train_Y_1), dim=0)
        
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

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=100,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(100, 25)

    def forward(self, x):

        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
rnn.cuda()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1, 28, 28).cuda().type(torch.cuda.FloatTensor)            # reshape x to (batch, time_step, input_size)
        b_y = b_y.cuda()
        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

    accuracy=0.0
    sum_data=0.0
    for valid_x,valid_y in valid_loader:
        valid_x=valid_x.view(-1, 28, 28).cuda().type(torch.cuda.FloatTensor)
        valid_y=valid_y.cuda() 
        valid_output = rnn(valid_x)
        pred_y = torch.max(valid_output, 1)[1].cuda().data  # move the computation in GPU

        accuracy += torch.sum(pred_y == valid_y).type(torch.FloatTensor) 
        sum_data+=valid_y.size(0)
    accuracy=accuracy/sum_data
    if accuracy>BEST_VALID_ACC:
        BEST_VALID_ACC=accuracy
    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| valid accuracy: %.4f' % accuracy)
print('Best valid accuracy: %.4f' %BEST_VALID_ACC)
