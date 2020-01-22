"""
    EXTERNAL LIBRARIES
"""

import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from torch import nn
import torch
import torch.nn.functional as F
from torch.utils import data

"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.auxfunctions import mapRange


"""
    It has been noted that the first layers of layered networks often 
    learn very slowly because the error derivatives are attenuated as they
    propagate back from the output layer toward the input.

    If no other information is available it might be assumed that the node
    outputs y are uniformly distributed on [0, 1] in which case the
    expected attenuation due to each sigmoid derivative is 1/6. 
    It can be suggested to scale the back-propagated derivatives by 6 to
    compensate. This would be equivalent to increasing the learning rate
    by 6 for weights into the last layer, by 36 for weights into the 
    second-to-last layer, 6^3 = 216 for weights 3 layers back from the
    output, and so on.
"""

# Define the 4-bit parity problem
def createAllBit4Strings():
    x, y = [], []
    maxValue = 2**4 - 1
    rangeValues = range(maxValue)
    for value in rangeValues:
        evenOnes = True
        binaryString = '{0:04b}'.format(value)
        binaryStringArray = list(binaryString)
        for i in range(len(binaryStringArray)):
            binaryStringArray[i] = int(binaryStringArray[i])
            if binaryStringArray[i] == 0:
                binaryStringArray[i] = -1
            if binaryStringArray[i] == 1:
                evenOnes = not evenOnes
        x.append(binaryStringArray)
        if evenOnes:
            y.append(0.9)
        else:
            y.append(-0.9)
    return np.asarray(x), np.asarray(y)

x, y = createAllBit4Strings()
tensor_x = torch.Tensor(x)
tensor_y = torch.Tensor(y)
dataset = data.TensorDataset(tensor_x, tensor_y)
dataloader = data.DataLoader(dataset)

# Create the Neural Network class via PyTorch
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):
        # define forward pass
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

# Define the loss function
criterion = nn.MSELoss()

# Create the two networks
normalNetwork = Network()
compensatedNetwork = Network()

# Define optimizers, one for normal learning rate, and one for delta
# attenuation compensation
import torch.optim as optim

lr = 0.001
normalOptimizer = optim.SGD(normalNetwork.parameters(), lr=lr, momentum=0.0)
compensatedOptimizer = optim.SGD([
    {'params': compensatedNetwork.fc4.parameters(), 'lr': lr*6**0},
    {'params': compensatedNetwork.fc3.parameters(), 'lr': lr*6**1},
    {'params': compensatedNetwork.fc2.parameters(), 'lr': lr*6**2},
    {'params': compensatedNetwork.fc1.parameters(), 'lr': lr*6**3}
], lr=lr, momentum=0.0)

"""
    Testing
"""

for epoch in range(1000):

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, targets = data

        # zero the parameter gradients
        compensatedOptimizer.zero_grad()

        outputs = compensatedNetwork(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        compensatedOptimizer.step()

        running_loss += loss.item()
    
    print(running_loss / 16)

print('Finished Training')