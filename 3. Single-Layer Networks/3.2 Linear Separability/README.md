## 3.2 Linear Separability

A single-layer perceptron can correctly classify only datasets which are <i>linearly separable</i>. Two classes are linearly separable only if a plane exists such that class <i>A</i> and <i>B</i> lie on opposite sides.

The exclusive-OR (XOR) function is a well-known example of a simple function that is not linearly separable and thus not computable by single-layer perceptrons.

For a comparison, let's train a single-layer network on both the AND function which is linearly separable and the XOR function which is not. They will be trained for a similar number of epochs, and the decision boundaries they have learned will be displayed every 2000 epochs. The graphic below shows the results:


![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.2%20Linear%20Separability/images/1.png)


As we can see, the network fails to learn the XOR function, but it learns the AND function very well. As the network trains on the AND network for more and more epochs, the decision boundary tightens up. This signals to us that the magnitude of the weight vector is increasing, while the orientation is staying roughly the same once it learns the plane that separates the points correctly.

14 of the 16 Boolean functions are linearly separable, and thus learnable by a single-layer network. We can view how a single-layer network trains and performs on all of these Boolean functions. The XOR and XNOR functions should be the only Boolean functions that the network fails to learn. The graphic below shows the results:


![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.2%20Linear%20Separability/images/2.png)


As we can see, the XOR and XNOR functions are indeed failing to be classified correctly. The rest of the functions have been classified correctly. Training for more epochs would serve to tighten up their decision boundaries.
