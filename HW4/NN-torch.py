
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader 
import torch.utils.data as data_utils


## torch nural network
class NeuralNetwork(nn.Module):
    def __init__(self, d1, d2, out_class=10):
        super(NeuralNetwork, self).__init__()
        self.nn_model = nn.Sequential(nn.Linear(28*28, d1),
                                   nn.Sigmoid(),
                                   nn.Linear(d1, d2), 
                                   nn.Sigmoid(),
                                   nn.Linear(d2, out_class), 
                                   nn.Softmax(dim=1),
                                  )
    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.nn_model(x)

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1457,), (0.2978,))])

## load the mnist dataset
root = './data/'
train_set = datasets.MNIST(root=root, train=True, transform=transforms, download=True)
test_set = datasets.MNIST(root=root, train=False, transform=transforms, download=True)

## to save time, use fisrt 10000 samples as traing set
# indices = torch.arange(0,10000)
# train_set = data_utils.Subset(train_set, indices)
# test_set = data_utils.Subset(test_set, indices)
# train_set = train_set[0:1000]
# test_set = train_set[0:100]

epochs = 30
batch_size = 64


train_subset, val_subset = random_split(
        train_set, [0.9, 0.1], generator=torch.Generator().manual_seed(1))

load_train = DataLoader(
                 dataset=train_subset,
                 batch_size=batch_size,
                 shuffle=True, drop_last=False)

load_val = DataLoader(
                 dataset=val_subset,
                 batch_size=batch_size,
                 shuffle=True, drop_last=False)

load_test = DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False, drop_last=False)



## start training
nn_model = NeuralNetwork(300, 200)
optimizer = optim.SGD(nn_model.parameters(), lr=1)
criterion = nn.CrossEntropyLoss(reduction='mean')

file = open("lcurve.txt", "w+")  

for i in range(epochs):
    train_loss = 0
    count = 0 
    for x_train, y_train in load_train:
        optimizer.zero_grad()
        loss = criterion(nn_model(x_train), y_train)
        train_loss += loss.data
        loss.backward()
        optimizer.step()
        count += 1
    train_loss /= count 
    

## start test
    val_loss = 0
    correct = 0
    total = 0 
    count = 0 
    with torch.no_grad():
        for x_test, y_test in load_test:
            loss = criterion(nn_model(x_test), y_test)
            val_loss += loss.data
            _, pred_label = torch.max(nn_model(x_test).data, 1)
            correct += (pred_label == y_test.data).sum()
            total += pred_label.shape[0]
            count += 1 
    
    val_loss /= count 
    error = 1 - correct * 1.0 / total
    # print('==>>> epoch: {}, train loss: {:.3f}'.format(epoch,  train_loss))
    print('epoch: {}/{}, error: {:.3f}'.format(
                i+1, epochs, error)) 
    error_save = error.cpu().detach().numpy()
    w1 = str(i+1)
    w2 = str(error_save)
    file.write(w1+' '+ w2+'\n') 
file.close()