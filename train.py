# Deep learning packages
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

# System packages
import os

# Custom packages
import dataclass
import config
import network


# Dataset preparation
testing_set = dataclass.testing_dataset(config.testing_csv_path,config.testing_img_path)
training_set = dataclass.training_dataset(config.training_csv_path, config.training_img_path)

train_loader = DataLoader(training_set, batch_size=config.param_batch_size, shuffle=True)
test_loader = DataLoader(testing_set,batch_size=config.param_batch_size,shuffle=True)

# Create a CNN
trail_net = network.CNN(config.input_channel,config.output_channel)

# CUDA support
if torch.cuda.is_available():       
    trail_net = trail_net.cuda()       

# Define Loss and Optimization 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(trail_net.parameters(), lr=config.param_learning_rate)

# Start Training
for epoch in range(config.num_epoches):
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
        output = trail_net(img)
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
