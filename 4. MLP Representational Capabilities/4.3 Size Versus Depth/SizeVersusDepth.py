"""
    EXTERNAL LIBRARIES
"""

import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import keras


"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.auxfunctions import plot_decision_boundary


"""
    Certain functions can be implemented exactly by small networks with
    two hidden layers but require an infinite number of nodes to
    approximate with a single hidden layer network.
"""

# Create a classifier that cannot be learned exactly by a
# one-hidden-layer network.
def classify(x, y):
    if abs(y) > abs(x):
        return 1
    return 0

def decision(x):
    return np.asarray([np.abs(x[:, 1]) > np.abs(x[:, 0])]).T

# Display the target decision boundary for the classifier function
fig, ax = plt.subplots(1, 1)

plot_decision_boundary(
    lambda x: decision(x),
    0, [-1, 1, -1, 1], ax, steppingSize=0.01
)

plt.show()

# Create the dataset based on the classifier function
train_x = []
train_y = []

for x in np.arange(-1.0, 1.0, 0.1):
    for y in np.arange(-1.0, 1.0, 0.1):
        train_x.append([x, y])
        train_y.append([classify(x, y)])

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

# Create optimizer for both networks
adam = keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Create single-hidden-layer network
singleHiddenNetwork = keras.models.Sequential()
singleHiddenNetwork.add(keras.layers.Dense(20, activation='sigmoid', input_shape=(2,)))
singleHiddenNetwork.add(keras.layers.Dense(1, activation='sigmoid'))

singleHiddenNetwork.compile(
    optimizer=adam, loss='mean_squared_error', metrics=['accuracy']
)

# Train the single-hidden-layer network
singleHiddenNetwork.fit(train_x, train_y, epochs=500)

# Plot the resulting decision boundary
fig, ax = plt.subplots(1, 1)

plot_decision_boundary(
    lambda x: singleHiddenNetwork.predict(x),
    0, [-1, 1, -1, 1], ax, steppingSize=0.01
)

plt.show()

# Create two-hidden-layer network
twoHiddenNetwork = keras.models.Sequential()
twoHiddenNetwork.add(keras.layers.Dense(10, activation='sigmoid', input_shape=(2,)))
twoHiddenNetwork.add(keras.layers.Dense(10, activation='sigmoid'))
twoHiddenNetwork.add(keras.layers.Dense(1, activation='sigmoid'))

twoHiddenNetwork.compile(
    optimizer=adam, loss='mean_squared_error', metrics=['accuracy']
)

# Train the two-hidden-layer network
twoHiddenNetwork.fit(train_x, train_y, epochs=500)

# Plot the resulting decision boundary
fig, ax = plt.subplots(1, 1)

plot_decision_boundary(
    lambda x: twoHiddenNetwork.predict(x),
    0, [-1, 1, -1, 1], ax, steppingSize=0.01
)

plt.show()