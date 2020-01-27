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
epochs = 1000

# Set the pytorch device to cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Define a simple classification problem
x, y = datasets.make_blobs(
    n_samples=100, n_features=2, centers=2, cluster_std=0.1,
    center_box=(-1.0, 1.0), shuffle=True, random_state=None
)

# Create PyTorch dataloader from the generated dataset
tensor_x = torch.from_numpy(x).float()
tensor_y = torch.from_numpy(y).float()
dataset = data.TensorDataset(tensor_x, tensor_y)
dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)

# Plot the dataset
fig, ax = plt.subplots(1, 1, dpi=60)

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
        tensor *= 2.0
        tensor -= 0.5 * tensor
    
# Create uniform random weight initialization function with values in the
# range (-2.4 / N, 2.4 / N), where N is the number of node inputs.
def randomInitNoVariable(tensor):
    tensorShape = list(tensor.shape)
    with torch.no_grad():
        tensor *= 0.0
        tensor += torch.rand(tensorShape) * 2.4 / tensorShape[1]
        tensor *= 2.0
        tensor -= 0.5 * tensor

# Create function that uses a formula for determining initial weight
# range, and using that to create a random uniform weight distribution
# Used for bipolar inputs x in {-1, +1}
def randomUniformCalculatedBipolar(tensor, probOne=0.5):
    tensorShape = list(tensor.shape)
    with torch.no_grad():
        tensor *= 0.0
        tensor += torch.rand(tensorShape) * 1.28
        tensor /= math.sqrt(tensorShape[1] * probOne * (1.0 - probOne))
        tensor *= 2.0
        tensor -= 0.5 * tensor

# Create function that uses a formula for determining initial weight
# range, and using that to create a random uniform weight distribution
# Used for binary inputs x in {0, 1}
def randomUniformCalculatedBinary(tensor, probOne=0.5):
    tensorShape = list(tensor.shape)
    with torch.no_grad():
        tensor *= 0.0
        tensor += torch.rand(tensorShape) * 2.55
        tensor /= math.sqrt(tensorShape[1] * probOne * (1.0 - probOne))
        tensor *= 2.0
        tensor -= 0.5 * tensor

# Create function that uses a formula for determining initial weight
# range, and using that to create a random uniform weight distribution
# Used for uniform inputs in the range [-a, +a]
def randomUniformCalculated(tensor, absRangeValue):
    tensorShape = list(tensor.shape)
    with torch.no_grad():
        tensor *= 0.0
        tensor += torch.rand(tensorShape) * 4.4
        tensor /= absRangeValue * math.sqrt(tensorShape[1])
        tensor *= 2.0
        tensor -= 0.5 * tensor

# Create function that uses a formula for determining initial weight
# range, and using that to create a random uniform weight distribution
# Used for Gaussian inputs N(0, sigma)
def randomGaussianCalculated(tensor, sigma):
    tensorShape = list(tensor.shape)
    with torch.no_grad():
        tensor *= 0.0
        tensor += torch.rand(tensorShape) * 2.55
        tensor /= sigma * math.sqrt(tensorShape[1])
        tensor *= 2.0
        tensor -= 0.5 * tensor

# Create function that uses a formula for determining initial weight
# distribution, and using that to create a random Gaussian weight
# distribution.
# Used for bipolar inputs x in {-1, +1}
def gaussianCalculatedBipolar(tensor, probOne=0.5):
    tensorShape = list(tensor.shape)
    with torch.no_grad():
        sigma = 0.74 / math.sqrt(tensorShape[1] * probOne * (1.0-probOne))
        tensor.normal_(mean=0, std=sigma)

# Create function that uses a formula for determining initial weight
# distribution, and using that to create a random Gaussian weight 
# distribution.
# Used for binary inputs x in {0, 1}
def gaussianCalculatedBinary(tensor, probOne=0.5):
    tensorShape = list(tensor.shape)
    with torch.no_grad():
        sigma = 1.47 / math.sqrt(tensorShape[1] * probOne * (1.0-probOne))
        tensor.normal_(mean=0, std=sigma)

# Create function that uses a formula for determining initial weight
# distribution, and using that to create a random Gaussian weight
# distribution.
# Used for uniform inputs in the range [-a, +a]
def gaussianCalculated(tensor, absRangeValue):
    tensorShape = list(tensor.shape)
    with torch.no_grad():
        sigma = 2.54 / (absRangeValue * math.sqrt(tensorShape[1]))
        tensor.normal_(mean=0, std=sigma)



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

    def initRandomUniformCalculatedBipolar(self):
        randomUniformCalculatedBipolar(self.fc1.weight)
        randomUniformCalculatedBipolar(self.fc2.weight)
        randomUniformCalculatedBipolar(self.fc3.weight)
    
    def initRandomUniformCalculatedBinary(self, probOne=0.5):
        randomUniformCalculatedBinary(self.fc1.weight, probOne)
        randomUniformCalculatedBinary(self.fc2.weight, probOne)
        randomUniformCalculatedBinary(self.fc3.weight, probOne)

    def initRandomUniformCalculated(self, absRangeValue):
        randomUniformCalculated(self.fc1.weight, absRangeValue)
        randomUniformCalculated(self.fc2.weight, absRangeValue)
        randomUniformCalculated(self.fc3.weight, absRangeValue)

    def initRandomGaussianCalculated(self, sigma):
        randomGaussianCalculated(self.fc1.weight, sigma)
        randomGaussianCalculated(self.fc2.weight, sigma)
        randomGaussianCalculated(self.fc3.weight, sigma)

    def initGaussianCalculatedBipolar(self, probOne=0.5):
        gaussianCalculatedBipolar(self.fc1.weight, probOne)
        gaussianCalculatedBipolar(self.fc2.weight, probOne)
        gaussianCalculatedBipolar(self.fc3.weight, probOne)

    def initGaussianCalculatedBinary(self, probOne=0.5):
        gaussianCalculatedBinary(self.fc1.weight, probOne)
        gaussianCalculatedBinary(self.fc2.weight, probOne=0.5)
        gaussianCalculatedBinary(self.fc3.weight, probOne=0.5)

    def initGaussianCalculated(self, absRangeValue):
        gaussianCalculated(self.fc1.weight, absRangeValue)
        gaussianCalculated(self.fc2.weight, absRangeValue)
        gaussianCalculated(self.fc3.weight, absRangeValue)


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
# methods. Train num_to_test networks on the problem for each weight initialization
# method, find the average loss, and save the data.
fig, ax = plt.subplots(1, 1, dpi=60)

num_to_test = 10

losses1 = []
for i in range(num_to_test):
    losses = np.zeros(epochs)
    network, optimizer = createNetwork()
    losses += trainNetwork(network, optimizer)
    losses1.append(losses[-1])
    print(str(i+1) + '/' + str(num_to_test))
print('Network 1/6 COMPLETE . . .\n\n\n')

losses2 = []
for i in range(num_to_test):
    losses = np.zeros(epochs)
    network, optimizer = createNetwork()
    network.to('cpu')
    network.initRandom(2)
    network.to(device)
    losses += trainNetwork(network, optimizer)
    losses2.append(losses[-1])
    print(str(i+1) + '/' + str(num_to_test))
print('Network 2/6 COMPLETE . . .\n\n\n')

losses3 = []
for i in range(num_to_test):
    losses = np.zeros(epochs)
    network, optimizer = createNetwork()
    network.to('cpu')
    network.initRandom(3)
    network.to(device)
    losses += trainNetwork(network, optimizer)
    losses3.append(losses[-1])
    print(str(i+1) + '/' + str(num_to_test))
print('Network 3/6 COMPLETE . . .\n\n\n')

losses4 = []
for i in range(num_to_test):
    losses = np.zeros(epochs)
    network, optimizer = createNetwork()
    network.to('cpu')
    network.initRandomNoVariable()
    network.to(device)
    losses += trainNetwork(network, optimizer)
    losses4.append(losses[-1])
    print(str(i+1) + '/' + str(num_to_test))
print('Network 4/6 COMPLETE . . .\n\n\n')

losses5 = []
for i in range(num_to_test):
    losses = np.zeros(epochs)
    network, optimizer = createNetwork()
    network.to('cpu')
    network.initRandomUniformCalculated(
        max(
            max(np.amax(x[:, 0]), abs(np.amin(x[:, 0]))),
            max(np.amax(x[:, 1]), abs(np.amin(x[:, 1])))
        )
    )
    network.to(device)
    losses += trainNetwork(network, optimizer)
    losses5.append(losses[-1])
    print(str(i+1) + '/' + str(num_to_test))
print('Network 5/6 COMPLETE . . .\n\n\n')

losses6 = []
for i in range(num_to_test):
    losses = np.zeros(epochs)
    network, optimizer = createNetwork()
    network.to('cpu')
    network.initGaussianCalculated(
        max(
            max(np.amax(x[:, 0]), abs(np.amin(x[:, 0]))),
            max(np.amax(x[:, 1]), abs(np.amin(x[:, 1])))
        )
    )
    network.to(device)
    losses += trainNetwork(network, optimizer)
    losses6.append(losses[-1])
    print(str(i+1) + '/' + str(num_to_test))
print('Network 6/6 COMPLETE . . .\n\n\n')

ax.boxplot(
    [losses1, losses2, losses3, losses4, losses5, losses6],
    labels=[
        'PyTorch Default\nInitialization',
        'Uniform Range\n(-A/sqrt(N), A/sqrt(N))\nA=2',
        'Uniform Range\n(-A/sqrt(N), A/sqrt(N))\nA=3',
        'Uniform Range\n(-2.4/N, 2.4/N)',
        'Uniform Calculated\nUniform Inputs',
        'Gaussian Calculated\nUniform Inputs'
    ]
)
ax.title.set_text('E(t) over Time with Various Weight Initialization Schemes')
ax.set_xlabel('Epochs')
ax.set_ylabel('E(t)\nMSE')

ax.legend()
plt.show()