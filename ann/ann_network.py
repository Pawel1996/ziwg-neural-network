import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
depth = 0

class Net(nn.Module):
    def __init__(self, count_start, count_neuron, layers_number):
        super().__init__()
        if layers_number < 2:
            LOG(" ### Siec musi miec co najmniej 2 warstwy ### ")
        self.input_layer = nn.Linear(count_start, count_neuron)
        self.hidden_layers = nn.ModuleList()
        for k in range(layers_number - 2):
            self.hidden_layers.append(nn.Linear(count_neuron, count_neuron))
        self.output_layer = nn.Linear(count_neuron, 10)

    def forward(self,x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return F.log_softmax(x,dim=1)

def LOG(message):
    print(" "*depth*2 + str(message))

