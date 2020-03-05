import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(pred_func, outputNum, graphDims, ax, valRange=(0.0, 1.0), steppingSize=0.01):
    """
        Plot decision boundary of a model as a contour plot
    """
    # Set min and max values and give it some padding
    x_min, x_max = graphDims[0] - .5, graphDims[1] + .5
    y_min, y_max = graphDims[2] - .5, graphDims[3] + .5
    h = steppingSize
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    print(Z.shape)
    Z = Z[:, outputNum]
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, vmin=valRange[0], vmax=valRange[1])