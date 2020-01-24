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
import torch.optim as optim

"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.auxfunctions import mapRange


"""
    Briefly, momentum has the following effects:

    It smooths weight changes by filtering out high frequency variations.
    When the learning rate is too high, momentum tends to suppress
    cross-stitching because consecutive opposing weight changes tend to
    cancel. The side to side oscillations across the valley damp out
    leaving only the components along the axis of the valley, which add up.

    When a long sequence of weight changes are all in the same direction,
    momentum tends to amplify the effective learning rate to η' = η/(1-α),
    leading to faster convergence.

    Momentum may sometimes help the system escape small local minima by
    giving the state vector enough inertia to coast over small bumps in the
    error surface.

    The following graphs produced demonstrate the effects of various
    momentum values on training.
"""

# Set the pytorch device to cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                binaryStringArray[i] = 0
            if binaryStringArray[i] == 1:
                evenOnes = not evenOnes
        x.append(binaryStringArray)
        if evenOnes:
            y.append(1.0)
        else:
            y.append(0.0)
    return np.asarray(x), np.asarray(y)

x, y = createAllBit4Strings()
tensor_x = torch.from_numpy(x).float()
tensor_y = torch.from_numpy(y).float()
dataset = data.TensorDataset(tensor_x, tensor_y)
dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True)

# Create the Neural Network class via PyTorch
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        # define forward pass
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create a network and optimizer given the momentum value
def createNetwork(momentumValue):
    lr = 10.0

    network = Network().to(device)
    optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentumValue)

    return network, optimizer

# Define the loss function
criterion = nn.MSELoss()

# Define the range of momentum values for both graphs
momentumsLow = [0.0, 0.2, 0.4, 0.6]
momentumsHigh = [0.6, 0.8, 0.9, 0.99]

# Create the network, train it on the 4-bit parity problem, and return the
# loss history
def getLosses(momentumValue, epochs):
    # Set up the losses array
    losses = []

    # Get the network and optimizer
    network, optimizer = createNetwork(momentumValue)
    network.train()

    # Train the network for the specified number of epochs
    for epoch in range(epochs):
        
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.view(-1, 1)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    
        losses.append(running_loss)
    
    return losses

for momentum in momentumsHigh:
    losses = getLosses(momentum, 3000)
    plt.plot(losses)
plt.show()