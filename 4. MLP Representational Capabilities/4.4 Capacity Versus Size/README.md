## 4.4 Capacity Versus Size

Another big question in designing a network is how many nodes to place in each layer. It is useful to have bounds on the
number of nodes that will be needed to fit a particular function.

After the network grows past a certain size, generalization criteria become the limiting factor on performance; a large
network can often fit the training data exactly, but it is unlikely to do so in a way that fits the underlying function that
generated the data.

If we have <i>m</i> training points, a network with a single layer of <i>m</i> - 1 hidden units can learn the data exactly,
since a line can always be found onto which the points project uniquely. Of course, this is inefficient and generalizes
very badly; it uses as much storage as a nearest neighbor classifier, takes about the same time to evaluate on a serial
computer, and probably generalizes worse.

In general, more efficient solutions are sought. Most interesting functions have structure, so each node should be able to
account for more than just one training sample.

A simple comparison between a single-hidden-layer network with 20 nodes in the hidden layer and a single-hidden-layer network
with M - 1 nodes in the hidden layer is shown below, trained on the same classification dataset.

The ground truth is shown below:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/4.%20MLP%20Representational%20Capabilities/4.4%20Capacity%20Versus%20Size/images/1.png)

The single-hidden-layer network with 20 nodes trained on this dataset for 250 epochs is shown below:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/4.%20MLP%20Representational%20Capabilities/4.4%20Capacity%20Versus%20Size/images/3.png)

The single-hidden-layer network with M - 1 nodes trained on this dataset for 250 epochs is shown below:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/4.%20MLP%20Representational%20Capabilities/4.4%20Capacity%20Versus%20Size/images/2.png)

The network with M - 1 nodes in its hidden layer is able to completely separate the given dataset correctly. However,
it appears to generalize very poorly -- which is to be expected. The network with only 20 nodes does not classify the
dataset with 100% accuracy, however it does appear to generalize better.
