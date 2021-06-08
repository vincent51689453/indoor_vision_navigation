# Deep learning packages
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

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
        out = self.conv(x)      
        out = out.view(out.size(0),-1)   
        out = self.fc(out)      
        return out