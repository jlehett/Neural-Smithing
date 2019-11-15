import numpy as np
import matplotlib.pyplot as plt

def frange(start, stop, step):
    """
        Range function that accepts floats.
        Will produce the range of numbers from start to stop (inclusive)
        with step value provided. If stop is not produced exactly, will
        only produce the value right before it (not over).
    """
    i = start
    _range = []
    while i <= stop:
        _range.append(i)
        i += step
    return _range

def plot_decision_boundary(pred_func, graphDims, ax):
    """
        Plot decision boundary of a model as a contour plot
    """
    # Set min and max values and give it some padding
    x_min, x_max = graphDims[0] - .5, graphDims[1] + .5
    y_min, y_max = graphDims[2] - .5, graphDims[3] + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu)

def getWeights(model):
    """ 
        Function to get the x and y weights and bias terms from single layer model in Keras.
    """
    weights = model.get_weights()[0]
    x = weights[0][0]
    y = weights[1][0]
    bias = model.get_weights()[1][0]
    return [x, y, bias]

def plotHyperplane(model, ax, target_mag, label):
    """
        Function to plot the hyperplane defined by the weights in a singler layer model in Keras.
    """
    weights = getWeights(model)
    
    # Grab a point with the constant sum of 0 with bias
    if weights[0] != 0:
        maxxcoord = target_mag
        maxycoord = (0 - weights[0] * maxxcoord - weights[2]) / weights[1]
    else:
        maxycoord = target_mag
        maxxcoord = (0 - weights[1] * maxycoord - weights[2]) / weights[0]
    if weights[0] != 0:
        minxcoord = -target_mag
        minycoord = (0 - weights[0] * minxcoord - weights[2]) / weights[1]
    else:
        minycoord = -target_mag
        minxcoord = (0 - weights[1] * minycoord - weights[2]) / weights[0]

    # Plot the hyperplane with the constant sum of 0 with bias
    ax.plot(
        [minxcoord, maxxcoord], [minycoord, maxycoord],
        linewidth=3, label=label, linestyle='--', color='black'
    )