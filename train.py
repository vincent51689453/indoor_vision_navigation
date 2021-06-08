import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
import os

import dataclass

testing_csv_path = './datasets/gazebo_ic382/test/test.csv'
testing_img_path = './datasets/gazebo_ic382/test/'

training_csv_path = './datasets/gazebo_ic382/train/train.csv'
training_img_path = './datasets/gazebo_ic382/train/'

param_batch_size = 8
param_learning_rate = 1e-2
num_epoches = 20

# Dataset preparation
testing_set = dataclass.testing_dataset(testing_csv_path,testing_img_path)
training_set = dataclass.training_dataset(training_csv_path, training_img_path)

train_loader = DataLoader(training_set, batch_size=param_batch_size, shuffle=True)
test_loader = DataLoader(testing_set,batch_size=param_batch_size,shuffle=True)

# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self,in_dim,n_class):
        super(CNN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim,6,kernel_size=3,stride=1,padding=1),   
            nn.ReLU(True),        
            nn.MaxPool2d(2,2),    
            nn.Conv2d(6,16,5,stride=1,padding=0), 
            nn.ReLU(True),
            nn.MaxPool2d(2,2)    
        )
        self.fc = nn.Sequential(  
            nn.Linear(576384,120),
            nn.Linear(120,84),
            nn.Linear(84,n_class)
        )
    def forward(self, x):
        out = self.conv(x)      #out shape(batch,16,5,5)
        out = out.view(out.size(0),-1)   #out shape(batch,400)
        out = self.fc(out)      #out shape(batch,10)
        return out

network = CNN(3,10)

if torch.cuda.is_available():       
     network = network.cuda()       

# Define Loss and Optimization 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=param_learning_rate)

# Start Training
for epoch in range(num_epoches):
    print('epoch{}'.format(epoch+1))
    print('*'*10)

    running_loss = 0.0
    running_acc = 0.0

    for i,data in enumerate(train_loader,1):
        img,label = data
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        img = img.float()
        output = network(img)
        loss = criterion(output,label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(output,1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        size_training_set = training_set.__len__()

        print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(epoch+1,running_loss/(size_training_set),\
            running_acc/size_training_set))
