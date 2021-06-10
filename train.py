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

# Training Loss and Acc
train_loss_list = []
train_acc_list = []

# Test Loss and Acc
test_loss_list = []
test_acc_list = []

# Start Training
for epoch in range(config.num_epoches):
    print('epoch{}'.format(epoch+1))
    print('*'*10)

    iteration = 0
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
        train_loss_list.append(loss.item())

        # Batch size
        total_sample_in_batch = label.size(0)

        # Getting the largetst P(x)
        _,predicted = torch.max(output.data,1)

        # Calculate number of matched labels, .item() convert the tensor to integer
        correct = (predicted == label).sum().item()
        train_acc_list.append(correct/total_sample_in_batch)

        # Make gradients parameters become zero
        optimizer.zero_grad()

        # Back propagation + optimization
        loss.backward()
        optimizer.step()

        # Print output
        print("Training -> Epoch[{}/{}],Iteration:{},Loss:{:.6f},Accuracy:{:.2f}%".format(          \
            epoch+1,config.num_epoches,i,loss.item(),((correct/total_sample_in_batch)*100)\
        ))

    # Write to tensorboard
    writer.add_scalar('Train/Loss',loss.item(),epoch)
    writer.add_scalar('Train/Accuracy',((correct/total_sample_in_batch)*100),epoch)
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

        # Loss evaluations
        loss = criterion(output,label)
        test_loss_list.append(loss.item())

        # Batch size
        total_sample_in_batch = label.size(0)

        # Getting the largetst P(x)
        _,predicted = torch.max(output.data,1)

        # Calculate number of matched labels, .item() convert the tensor to integer
        correct = (predicted == label).sum().item()
        test_acc_list.append(correct/total_sample_in_batch)

        # Make gradients parameters become zero
        optimizer.zero_grad()

        # Back propagation + optimization
        loss.backward()
        optimizer.step()

        # Print output
        print("Testing -> Epoch[{}/{}],Iteration:{},Loss:{:.6f},Accuracy:{:.2f}%".format(          \
            epoch+1,config.num_epoches,i,loss.item(),((correct/total_sample_in_batch)*100)\
        ))

    # Write to tensorboard
    writer.add_scalar('Test/Loss',loss.item(),epoch)
    writer.add_scalar('Test/Accuracy',((correct/total_sample_in_batch)*100),epoch)
    writer.flush()

# Save checkpoint
torch.save(navigation_net,'models/navigation_net_v999.pt')
print("Training is over")