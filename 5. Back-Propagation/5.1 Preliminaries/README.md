## 5.1 Preliminaries

### Forward Propagation

Back-propagation can be applied to any feedforward network with differentiable activation functions. In particular, it is not 
necessary that it have a layered structure.

Assume nodes are indexed so that <i>i</i> > <i>j</i> implies that node <i>i</i> follows node <i>j</i> in terms of dependency.
But node <i>j</i> < <i>i</i> does not depend on node <i>i</i>.

In the forward pass, the network computes an output based on its current inputs. Each node <i>i</i> computes a weighted sum
<i>a<sub>i</sub></i> of its inputs and passes this through a nonlinearity to obtain the node output <i>y<sub>i</sub></i>.

<i>a<sub>i</sub></i> = <b>SUM</b><sub><i> j</i> < <i>i</i></sub> [ <i>w<sub>ij</sub> y<sub>j</sub></i> ]

<i>y<sub>i</sub></i> = <i>f</i>(<i>a<sub>i</sub></i>)

Normally <i>f</i> is a bounded monotonic function such as tanh or sigmoid. Arbitrary differentiable functions can be used, but
sigmoid-like "squashing" functions are standard. The index <i>j</i> in the sum runs over all indexes <i>j</i> < <i>i</i> of
nodes that could send input to node <i>i</i>.

Every node is evaluated in order, starting with the first hidden node and continuing to the last output node. In layered
networks, the first hidden layer is updated based on the external inputs, the second hidden layer is updated based on the
outputs of the first hidden layer, and so on to the output layer which is updated based on the outputs of the last hidden
layer. At the end of the sweep, the system outputs will be available at the output nodes.

### Error Calculation

Unless the network is perfectly trained, the network outputs will differ somewhat from the desired outputs. The significance
of these differences is measured by an error (or cost) function <i>E</i>. One possible error function to use is the SSE
errr function:

<i>E</i> = <sup>1</sup>/<sub>2</sub> * <b>SUM</b> <sub><i>p</i></sub> [ <b>SUM</b> <sub><i>i</i></sub> [ (<i>d<sub>pi</sub>
</i> - <i>y<sub>pi</sub></i> ) <sup>2</sup> ] ]

where <i>p</i> indexes the patterns in the training set, <i>i</i> indexes the output nodes, and <i>d<sub>pi</sub></i> and
<i>y<sub>pi</sub></i> are, respectively, the desired target and actual network output for the <i>i</i>th output node on the
<i>p</i>th pattern. The <sup>1</sup>/<sub>2</sub> factor suppresses a factor of 2 later on. One of the reasons that SSE is
convenient is that errors on different patterns and different outputs are independent; the overall error is just the sum of
the individual squared errors.

<i>E</i> = <b>SUM</b> <sub><i>p</i></sub> [ <i>E<sub>p</sub></i> ]

<i>E<sub>p</sub></i> = <sup>1</sup>/<sub>2</sub> * <b>SUM</b> <sub><i>i</i></sub> [ (<i>d<sub>pi</sub></i> -
<i>y<sub>pi</sub></i>)<sup>2</sup> ]

The code in this section demonstrates how to perform a forward pass and calculate the SSE of the network given target
outputs.

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/5.%20Back-Propagation/5.1%20Preliminaries/images/1.png)

### [Continue to Section 5.2](https://github.com/jlehett/Neural-Smithing/tree/master/5.%20Back-Propagation/5.2%20Back-Propagation:%20The%20Derivative%20Calculation)
