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
from sklearn.decomposition import PCA

"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.auxfunctions import mapRange


"""
    Principal components analysis (PCA) attempts to identify the 
    major axes of variation of a data set -- the directions along 
    which the data varies the most. 
    
    Going on the assumption that these components are important 
    directions for the function to be learned, one can initialize 
    the input-to-hidden weight vectors along these directions. In a
    network with H hidden nodes the H eigenvectors with the largest
    eigenvalues would be selected.
    
    In the first phase of training, the input-to-hidden weights
    are fixed to the principal component directions while the 
    output weights are trained. In the second phase, all weights 
    are allowed to learn.
"""

# Set the pytorch device to cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Create the Neural Network class via PyTorch
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        # Define forward pass
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def initPCA(self):
        # Initialize the first set of weights based on the PCA eigenvectors
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.from_numpy(pca.components_).float())
        print(self.fc1.weight)


# Define the loss function
criterion = nn.MSELoss()

# Define a function to create the network
import torch.optim as optim

def createNetwork():
    network = Network().to(device)
    optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.0)
    return network, optimizer

def createPCANetwork():
    network = Network()
    network.initPCA()
    network = network.to(device)
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

def trainPCANetwork(network, optimizer):
    network.train()
    # Phase 1
    network.fc1.weight.requires_grad = False
    losses = []
    for epoch in range(round(epochs/2)):

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
    
    # Phase 2
    network.fc1.weight.requires_grad = True
    for epoch in range(round(epochs/2)):
        
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

# Train 2 networks, one with the weight initialized according to PyTorch defaults,
# and one initialized with PCA and trained in the two-phase algorithm.
fig, ax = plt.subplots(1, 1, dpi=60)

num_to_test = 10
epochs = 1000
lr = 0.1
batch_size = 16

defLosses = []
pcaLosses = []
for i in range(num_to_test):
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

    # Perform PCA on the dataset
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    losses = np.zeros(epochs)
    defNetwork, defOptimizer = createNetwork()
    losses += trainNetwork(defNetwork, defOptimizer)
    defLosses.append(losses[-1])

    losses = np.zeros(epochs)
    pcaNetwork, pcaOptimizer = createPCANetwork()
    losses += trainPCANetwork(pcaNetwork, pcaOptimizer)
    pcaLosses.append(losses[-1])

    print(str(i+1) + '/' + str(num_to_test))

ax.boxplot(
    [defLosses, pcaLosses],
    labels=[
        'PyTorch Default\nInitialization',
        'PCA Initialization'
    ]
)
ax.title.set_text('E(t) over Time for Different Weight Initialization Schemes')
ax.set_xlabel('Epochs')
ax.set_ylabel('E(t)\nMSE')

ax.legend()
plt.show()