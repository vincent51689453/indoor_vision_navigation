# Deep learning packages
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter 

# System packages
import os

# Custom packages
import dataclass
import config
import network

# Tensorboard writer
writer = SummaryWriter('./log')

# Dataset preparation
testing_set = dataclass.testing_dataset(config.testing_csv_path,config.testing_img_path)
training_set = dataclass.training_dataset(config.training_csv_path, config.training_img_path)

train_loader = DataLoader(training_set, batch_size=config.param_batch_size, shuffle=True)
test_loader = DataLoader(testing_set,batch_size=config.param_batch_size,shuffle=True)

# Create a CNN
navigation_net = network.CNN(config.input_channel,config.output_channel)

# CUDA support
if torch.cuda.is_available():       
    navigation_net = navigation_net.cuda()       

# Define Loss and Optimization 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(navigation_net.parameters(), lr=config.param_learning_rate)

# Start Training
for epoch in range(config.num_epoches):
    print('epoch{}'.format(epoch+1))
    print('*'*10)

    running_loss = 0.0
    running_acc = 0.0

    # Training at each epoch
    for i,data in enumerate(train_loader,1):
        img,label = data
     
        #Normalization
        img = img/255

        # Add CUDA support
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        # Network IN/OUT
        img = img.float()
        output = navigation_net(img)

        # Loss evaluations
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

        scalar_loss = running_loss/size_training_set
        scalar_accuracy = running_acc/size_training_set

        print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(epoch+1,scalar_loss,scalar_accuracy))
        # Write to tensorboard
        writer.add_scalar('Train/Loss',scalar_loss,epoch)
        writer.add_scalar('Train/Accuracy',scalar_accuracy,epoch)
        writer.flush()


    # Testing at each epoch

    # Freeze network at instant
    navigation_net.eval()
    
    eval_loss = 0
    eval_acc = 0
    for i,data in enumerate(test_loader,1):
        img, label = data

        #Normalization
        img = img/255

        # Add CUDA support
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        # Network IN/OUT
        img = img.float()
        output = navigation_net(img)
        loss = criterion(output,label)

        eval_loss += loss.item()*label.size(0)
        _, pred = torch.max(output,1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        eval_acc += num_correct.item()

        size_testing_set = testing_set.__len__()
        scalar_loss = eval_loss/size_testing_set
        scalar_accuracy = eval_acc/size_testing_set

        print('Test Loss: {:.6f}, Acc: {:.6f}\r\n'.format(scalar_loss,scalar_accuracy))
        # Write to tensorboard
        writer.add_scalar('Test/Loss',scalar_loss,epoch)
        writer.add_scalar('Test/Accuracy',scalar_accuracy,epoch)
        writer.flush()
