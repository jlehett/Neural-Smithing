"""
    EXTERNAL LIBRARIES
"""

import sys
import math
import matplotlib.pyplot as plt
import numpy as np


"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.auxfunctions import frange


"""
    The capacity of a hyperplane is the number of dichotomies
    it can separate. For N points in general position in a Euclidean
    space of dimension d, the number C(N, d) of homogeneously linearly
    separable dichotomies can be found.

    If the N points are not in general position, the number of linearly
    separable dichotomies may be much lower. Requirement of 
    general position:
        For N > d, no set of d+1 points lie on a (d-1)-dimensional hyperplane
        For N <= d, no (d-2)-dimensional hyperplane contains them all.
    
    A set of points is homogeneously linearly separable if the points
    can be separated by a hyperplane passing through the origin. A linear
    separation of N points in d dimensions is a homogeneous linear separation
    in d+1 dimesnions because the offset of a hyperplane that does not pass
    through the origin can be absorbed into an extra bias weight.
"""

print('\n\nEnter values to solve for the capacity of the hyperplane ' +
      'given N points in d dimensions.')

# Grab user input of number of points
N = int(input('\nEnter number of points, N: '))

# Grab user input of number of dimensions
d = int(input('Enter number of dimensions, d: '))

# Function to find number of homogeneously linearly separable dichotomies
def C(N, d):
    if N <= d + 1:
        return 2**N
    sum = 0
    for k in range(0, d+1):
        sum += math.factorial(N-1) // (math.factorial(k) * math.factorial(N-1-k))
    return 2 * sum

# Print Result
print('\nThe capacity of a hyperplane separating ' +
      'N points in d dimensions = ' + str(C(N, d)))


"""
    All dichotomies on N <= d + 1 points in general position are linearly
    separable in d dimensions. For N > d + 1, only C(N, d) of the 2^N
    possibe dichotomies are linearly separable. We can find the probability
    that a randomly chosen dichotomy is linearly separable.
"""

print('\n\nEnter values to solve for the probability that a randomly ' +
      'chosen dichotomy is linearly separable given N points in ' +
      'd dimensions.')

# Grab user input
N = int(input('\n\nEnter number of points, N: '))
d = int(input('Enter number of dimensions, d: '))

# Function to find the probability that a randomly chosen dichotomy is
# linearly separable
def f(N, d):
    return C(N, d) / 2**N

# Find the probability that a randomly chosen dichotomy is linearly separable
# for user input
print('\nThe probability that a randomly chosen dichotomy is linearly ' +
      'separable for N points in d dimensions is {0:.2f}'.format(f(N, d)))


"""
    Graph of the probability that a randomly chosen dichotomy is linearly
    separable for various values of d.
"""

# Create the plot
fig, ax = plt.subplots(1, 1)
fig.suptitle('Probability that a randomly chosen dichotomy is\n' +
             'linearly separable for various values of d', fontsize=18)

# Get the range of x values
x = [_ for _ in frange(0, 4.1, 0.1)]

# Set up containers for plotting
valuesOfD = [1, 5, 10, 25, 50, 100, 200]
ySets = [[] for i in range(len(valuesOfD))]

# For each value of x, find the corresponding probability for each value
# of d and add to ySets
for xtick in x:
    for dIndex in range(len(valuesOfD)):
        d = valuesOfD[dIndex]
        # Find the corresponding value of N
        N = round(xtick * (d + 1))
        # Find the corresponding probability
        P = f(N, d)
        # Add the probability to the appropriate ySets
        ySets[dIndex].append(P)

# Plot every line for each value of d
for dIndex in range(len(valuesOfD)):
    ax.plot(x, ySets[dIndex], label='d = ' + str(valuesOfD[dIndex]))

# Adjust graph parameters
ax.grid(linewidth=1, alpha=0.5)
ax.set_xlim(-0.3, 4.3)
ax.set_ylim(-0.3, 1.3)
ax.set_yticks(np.arange(-0.25, 1.5, 0.25))

# Set legend for the plot
fig.legend(loc='upper right', borderaxespad=1.0, fontsize=15)

# Set axis labels
ax.set_xlabel('N / (d + 1)', fontsize=18)
ax.set_ylabel('P', fontsize=18, rotation=0)

# Set caption
textstr = str('In linear regression, a common heuristic its to require\n' +
          'N ≥ 3d or more training patterns. Otherwise, there is a\n' +
          'high chance the network will learn a random dichotomy\n' +
          '(overfitting).')
props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
ax.text(2.25, 0.75, textstr, fontsize=14, bbox=props)

# Show graphs
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
