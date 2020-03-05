## 4.1 MLP Representational Capability

The standard multilayer perceptron (MLP) is a cascade of single-layer perceptrons. There is a layer of input nodes, a layer of output nodes, and one or more intermediate layers. The interior layers are sometimes called "hidden layers" because they are not directly observable from the system inputs and outputs.

Nodes in each layer are fully connected to nodes in the preceding and following layers. There are no connections between units in the same layer, connections from one layer back to a previous layer, or "shortcut" connections that skip over intermediate layers.

The representational capability of a network can be defined as the range of mappings it can implement when the weights are varied. A particular mapping is within the representational capability of a net if there is a set of weights for which the net performs the mapping.

When designing a layered network, an obvious first question is how many layers to use. Two hidden layers are sufficient to create classification regions of any desired shape. In the following diagram, linear threshold units in the first hidden layer divide the input space into half-spaces with hyperplanes, units in the second hidden layer AND (form intersections of) these half-spaces to produce convex regions, and the output units OR (form unions of) the convex regions into arbitrary, possibly unconnected, shapes. 

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/4.%20MLP%20Representational%20Capabilities/4.1%20Representational%20Capability/images/1.png)

Given a sufficient number of units, a network can be formed that divides the input space in any way desired, producing a 0 when the input is in one region and 1 when it is in another. The boundaries are piecewise linear, but any smooth boundary can be approximated with enough units.

To approximate continous functions, one can add and subtract (rather than logically OR) convex regions with appropriate weighting factors so two hidden layers are also sufficient to approximate any desired bounded continuous function.

It is sometimes mistakenly assumed that single-hidden-layer networks can recognize only convex decision regions. Counter examples are shown below:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/4.%20MLP%20Representational%20Capabilities/4.1%20Representational%20Capability/images/2.png)

In the above graph, a 'C' region is produced using only a single-hidden-layer network. This proves that a single-hidden-network is capable of producing nonconvex decision regions.

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/4.%20MLP%20Representational%20Capabilities/4.1%20Representational%20Capability/images/3.png)

In the above graph, a disjoint region is produced via a single-hidden-layer network. This proves that a single-hidden-network is also capable of producing disjoint regions.


### [Continue to Section 4.2](https://github.com/jlehett/Neural-Smithing/tree/master/4.%20MLP%20Representational%20Capabilities/4.2%20Universal%20Approximation%20Capabilities)
