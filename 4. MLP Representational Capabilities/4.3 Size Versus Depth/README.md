## 4.3 Size Versus Depth

Although a single hidden layer is optimal foor some functions, there are others for which a single-hidden-layer solution is
very inefficient compared to solutions with more layers. Certain functions can be implemented exactly by small networks with
two hidden layers but require an infinite number of nodes to approximate with a single hidden layer network.

An example of a problem which is solved much easier with 2 hidden layers compared to one can be seen below:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/4.%20MLP%20Representational%20Capabilities/4.3%20Size%20Versus%20Depth/images/1.png)

Here we see the ground truth for the classification problem. We will attempt to train a single hidden layer network with 42 weights on the problem, as well as a two hidden layer network with only 40 weights on the problem.

Below is the results of training the single hidden layer network with 42 weights for 500 epochs on the problem:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/4.%20MLP%20Representational%20Capabilities/4.3%20Size%20Versus%20Depth/images/2.png)

And here are the results of training the two hidden layer network with only 40 weights for 500 epochs on the problem:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/4.%20MLP%20Representational%20Capabilities/4.3%20Size%20Versus%20Depth/images/3.png)

The network with two hidden layers obtained a much higher accuracy than the network with only a single hidden layer, despite the fact that it is using fewer weights.

Of course, there are still functions where small single-hidden-layer networks are optimal and additional hidden layers are not useful. Single-hidden-layer networks may need large numbers of nodes to compute arbitrary functions, but small networks may suffice for particular functions. For example, in some studies, the one-hidden-layer nets were reported to have lower average error and perhaps generalized better.
