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
    If we have m training points, a network with a single layer
    of m - 1 hidden units can learn the data exactly, since a
    line can always be found onto which the points project
    uniquely.
"""
# Parameters
NUM_TRAINING_POINTS = 100

# Create a classifier that cannot be learned exactly by a
# one-hidden-layer network.
def classify(x, y):
    if abs(x) > abs(y):
        return 1
    return 0

def decision(x):
    return np.asarray([np.abs(x[:, 0]) > np.abs(x[:, 1])]).T

# Display the target decision boundary for the classifier function
fig, ax = plt.subplots(1, 1)

plot_decision_boundary(
    lambda x: decision(x),
    0, [-1, 1, -1, 1], ax, steppingSize=0.01
)

plt.title('Ground Truth Decision Boundary', fontsize=18)
plt.show()

# Create the dataset based on the classifier function
train_x = []
train_y = []

for i in range(NUM_TRAINING_POINTS):
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    train_x.append([x, y])
    train_y.append([classify(x, y)])

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

# Create optimizer for overfitting network
adam = keras.optimizers.Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Create overitting single-hidden-layer network
singleHiddenNetwork = keras.models.Sequential()
singleHiddenNetwork.add(keras.layers.Dense(NUM_TRAINING_POINTS-1, activation='hard_sigmoid', input_shape=(2,)))
singleHiddenNetwork.add(keras.layers.Dense(1, activation='hard_sigmoid'))

singleHiddenNetwork.compile(
    optimizer=adam, loss='mean_squared_error', metrics=['accuracy']
)

# Train the single-hidden-layer network
singleHiddenNetwork.fit(train_x, train_y, epochs=250)

# Plot the resulting decision boundary
fig, ax = plt.subplots(1, 1)

plot_decision_boundary(
    lambda x: singleHiddenNetwork.predict(x),
    0, [-1, 1, -1, 1], ax, steppingSize=0.01
)

# Scatter plot the points used for training to observe overfitting.
true_x = train_x[np.where(train_y == 1), :][0]
false_x = train_x[np.where(train_y == 0), :][0]

for pair in true_x:
    ax.scatter(pair[0], pair[1], color='b')
for pair in false_x:
    ax.scatter(pair[0], pair[1], color='r')

plt.title('Single-Hidden-Layer Network @ 250 Epochs\n' +
          '(M-1 nodes in Hidden Layer)', fontsize=18)
plt.show()

# Create optimizer for ok network
adam = keras.optimizers.Adam(learning_rate=0.02, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Create ok single-hidden-layer network
singleHiddenNetwork = keras.models.Sequential()
singleHiddenNetwork.add(keras.layers.Dense(20, activation='sigmoid', input_shape=(2,)))
singleHiddenNetwork.add(keras.layers.Dense(1, activation='sigmoid'))

singleHiddenNetwork.compile(
    optimizer=adam, loss='mean_squared_error', metrics=['accuracy']
)

# Train the single-hidden-layer network
singleHiddenNetwork.fit(train_x, train_y, epochs=250)

# Plot the resulting decision boundary
fig, ax = plt.subplots(1, 1)

plot_decision_boundary(
    lambda x: singleHiddenNetwork.predict(x),
    0, [-1, 1, -1, 1], ax, steppingSize=0.01
)

# Scatter plot the points used for training to observe overfitting.
true_x = train_x[np.where(train_y == 1), :][0]
false_x = train_x[np.where(train_y == 0), :][0]

for pair in true_x:
    ax.scatter(pair[0], pair[1], color='b')
for pair in false_x:
    ax.scatter(pair[0], pair[1], color='r')

plt.title('Single-Hidden-Layer Network @ 250 Epochs\n' +
          '(20 nodes in Hidden Layer)', fontsize=18)
plt.show()