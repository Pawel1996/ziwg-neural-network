import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
depth = 0

class Net(nn.Module):
    def __init__(self, count_start, count_neuron):
        super().__init__()
        self.fc1 = nn.Linear(count_start,count_neuron)
        self.fc2 = nn.Linear(count_neuron, count_neuron)
        self.fc3 = nn.Linear(count_neuron,9)

    def forward(self,x):
        x = F.relu(self.fc1(x) )
        x = F.relu(self.fc2(x) )
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

def LOG(message):
    print(" "*depth*2 + str(message))
