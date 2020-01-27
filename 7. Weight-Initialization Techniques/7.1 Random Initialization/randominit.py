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
from sklearn import datasets

"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.auxfunctions import mapRange


"""
    The normal initialization procedure is to set weights to "small" random
    values. The randomness is intended to break symmetry while small weights
    are chosen to avoid immediate saturation.

    Typically, weights are randomly selected from a range such as
    ( -A / sqrt(N), A / sqrt(N) ), where N is the number of inputs to the
    node and A is a constant between 2 and 3.

    The range (-2.4 / N, 2.4 / N), where N is the number of node inputs, is
    another commonly cited choice.
"""

# Set some parameters for the networks
batch_size = 16
lr = 0.1
epochs = 200

# Set the pytorch device to cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Define a simple classification problem
x, y = datasets.make_blobs(
    n_samples=50, n_features=2, centers=2, cluster_std=1.0,
    center_box=(-10.0, 10.0), shuffle=True, random_state=None
)

# Create PyTorch dataloader from the generated dataset
tensor_x = torch.from_numpy(x).float()
tensor_y = torch.from_numpy(y).float()
dataset = data.TensorDataset(tensor_x, tensor_y)
dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)

# Plot the dataset
fig, ax = plt.subplots(1, 1)

ax.scatter(x[:,0], x[:,1], c=y, s=25, marker='o')
ax.title.set_text('Generated Classification Problem')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

plt.show()

# Create uniform random weight initialization function with values in the
# range (-A / sqrt(N), A / sqrt(N)), where N is the number of inputs to
# the node and A is a constant between 2 and 3.
def randomInit(A, tensor):
    tensorShape = list(tensor.shape)
    with torch.no_grad():
        tensor *= 0.0
        tensor += torch.rand(tensorShape) * A / math.sqrt(tensorShape[1])
    
# Create uniform random weight initialization function with values in the
# range (-2.4 / N, 2.4 / N), where N is the number of node inputs.
def randomInitNoVariable(tensor):
    tensorShape = list(tensor.shape)
    with torch.no_grad():
        tensor *= 0.0
        tensor += torch.rand(tensorShape) * 2.4 / tensorShape[1]


# Create the Neural Network class via PyTorch
class Network(nn.Module):
    def __init__(self, weightInit='Default'):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        # define forward pass
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def initRandom(self, A):
        randomInit(A, self.fc1.weight)
        randomInit(A, self.fc2.weight)
        randomInit(A, self.fc3.weight)

    def initRandomNoVariable(self):
        randomInitNoVariable(self.fc1.weight)
        randomInitNoVariable(self.fc2.weight)
        randomInitNoVariable(self.fc3.weight)


# Define the loss function
criterion = nn.MSELoss()

# Define a function to create the network
import torch.optim as optim

def createNetwork():
    network = Network().to(device)
    optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.0)
    return network, optimizer

# Define a function that trains the network and returns loss history
def trainNetwork(network, optimizer):
    network.train()
    losses = []
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
    return np.asarray(losses)

# Train 4 networks, each with one of the different weight initialization
# methods. Train 10 networks on the problem for each weight initialization
# method, find the average loss, and save the data.
fig, ax = plt.subplots(1, 1)

num_to_test = 2000

for i in range(num_to_test):
    losses1 = np.zeros(epochs)
    network, optimizer = createNetwork()
    losses1 += trainNetwork(network, optimizer)
    print(str(i+1) + '/' + str(num_to_test))
losses1 /= num_to_test
ax.plot(losses1, label='PyTorch Default Initialization')
print('Network 1/4 COMPLETE . . .\n\n\n')

for i in range(num_to_test):
    losses2 = np.zeros(epochs)
    network, optimizer = createNetwork()
    network.to('cpu')
    network.initRandom(2)
    network.to(device)
    losses2 += trainNetwork(network, optimizer)
    print(str(i+1) + '/' + str(num_to_test))
losses2 /= num_to_test
ax.plot(losses2, label='Uniform Range (-A/sqrt(N), A/sqrt(N))\nA=2')
print('Network 2/4 COMPLETE . . .\n\n\n')

for i in range(num_to_test):
    losses3 = np.zeros(epochs)
    network, optimizer = createNetwork()
    network.to('cpu')
    network.initRandom(3)
    network.to(device)
    losses3 += trainNetwork(network, optimizer)
    print(str(i+1) + '/' + str(num_to_test))
losses3 /= num_to_test
ax.plot(losses3, label='Uniform Range (-A/sqrt(N), A/sqrt(N))\nA=3')
print('Network 3/4 COMPLETE . . .\n\n\n')

for i in range(num_to_test):
    losses4 = np.zeros(epochs)
    network, optimizer = createNetwork()
    network.to('cpu')
    network.initRandomNoVariable()
    network.to(device)
    losses4 += trainNetwork(network, optimizer)
    print(str(i+1) + '/' + str(num_to_test))
losses4 /= num_to_test
ax.plot(losses4, label='Uniform Range (-2.4/N, 2.4/N)')
print('Network 4/4 COMPLETE . . .\n\n\n')

ax.title.set_text('E(t) over Time with Various Weight Initialization Schemes')
ax.set_xlabel('Epochs')
ax.set_ylabel('E(t)\nMSE')

ax.legend()
plt.show()