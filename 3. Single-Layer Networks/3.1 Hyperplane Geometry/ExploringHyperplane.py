"""
    EXTERNAL LIBRARIES
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from math import sqrt


#plt.style.use('dark_background')

"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.SingleLayerNetwork import SingleLayerNetwork
from utils.auxfunctions import frange


"""
    The locus of points x with a constant sum
        SUM( Wj * Xj )
    defines a hyperplane perpendicular to the vector w.

    2D visualization
"""
# Params
TARGET_MAG = 5      # The value to scale the weights by for plotting
NUM_EXAMPLE_COLS = 4    # The number of rows of examples to plot
NUM_EXAMPLE_ROWS = 3    # The number of cols of examples to plot

# Define an activation function that does not change the input value.
# No activation function is needed for this visualization.
def noActivation(x):
    return x

# Create the pyplot
fig, axes = plt.subplots(NUM_EXAMPLE_ROWS, NUM_EXAMPLE_COLS)
fig.suptitle('The locus of points with a constant sum' +
             ' Σ( Wj * Xj ) defines\na hyperplane perpendicular' +
             ' to the vector w.', fontsize=18)

# Create each subplot as an example
weightVector, constantSumVector = None, None
for x in range(NUM_EXAMPLE_ROWS):
    for y in range(NUM_EXAMPLE_COLS):
        # Create Single Layer Network with randomized starting weights
        slp = SingleLayerNetwork(2, 1, noActivation, randomize=True)

        # Scale the W vector such that it has a magnitude of TARGET_MAG
        # and it is centered at (0, 0)
        slpWeights = slp.getWeights()[0]

        scaledWeightX = [
            -sqrt(TARGET_MAG**2 / (slpWeights[0]**2 + slpWeights[1]**2)) * slpWeights[0],
            sqrt(TARGET_MAG**2 / (slpWeights[0]**2 + slpWeights[1]**2)) * slpWeights[0]
        ]
        scaledWeightY = [
            -sqrt(TARGET_MAG**2 / (slpWeights[0]**2 + slpWeights[1]**2)) * slpWeights[1],
            sqrt(TARGET_MAG**2 / (slpWeights[0]**2 + slpWeights[1]**2)) * slpWeights[1]
        ]

        # Plot this weight vector as a blue arrow
        weightVector = axes[x, y].arrow(scaledWeightX[0], scaledWeightY[0], 
                scaledWeightX[1]*2, scaledWeightY[1]*2, 
                head_width=0.5, overhang=0.2, linestyle='-', color='blue',
                linewidth=5)

        # Grab a point with the constant sum of 0
        if slpWeights[1] != 0:
            xcoord = 1
            ycoord = (0 - slpWeights[0] * xcoord) / slpWeights[1]
        else:
            ycoord = 1
            xcoord = (0 - slpWeights[1] * ycoord) / slpWeights[0]

        # Form a line given this point, scaled to TARGET_MAG
        scaledPointX = [
            -sqrt(TARGET_MAG**2 / (xcoord**2 + ycoord**2)) * xcoord,
            sqrt(TARGET_MAG**2 / (xcoord**2 + ycoord**2)) * xcoord
        ]
        scaledPointY = [
            -sqrt(TARGET_MAG**2 / (xcoord**2 + ycoord**2)) * ycoord,
            sqrt(TARGET_MAG**2 / (xcoord**2 + ycoord**2)) * ycoord
        ]

        # Plot the locus of points with constant sum of 0 as a red arrow
        constantSumVector = axes[x, y].arrow(scaledPointX[0], scaledPointY[0], 
                scaledPointX[1]*2, scaledPointY[1]*2,
                head_width=0.5, overhang=0.2, linestyle='-', color='red',
                linewidth=3)

        # Limit the x and y axis such that you get a square plot (no squashing)
        # Squashing would cause the line to look as though it weren't perpendicular.
        axes[x, y].set_xlim(-TARGET_MAG*1.3, TARGET_MAG*1.3)
        axes[x, y].set_ylim(-TARGET_MAG*1.3, TARGET_MAG*1.3)
        axes[x, y].set_aspect('equal', adjustable='box')

# Set global legend for the plot
fig.legend([weightVector, constantSumVector],
           ['Weight Vector', 'Points with constant sum of 0'],
           loc='upper right', borderaxespad=1.0, fontsize=15)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()