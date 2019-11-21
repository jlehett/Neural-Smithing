## 3.1 Hyperplane Geometry

The output of a single-layer network can be seen as:

<i>y</i>(<b>x</b>) = <i>f</i> (<b>w<sup>T</sup>x</b>)

where <b>x</b> and <b>w</b> are column vectors with elements <i>x<sub>j</sub></i> and <i>w<sub>j</sub></i>, and the superscript 
<i><b>T</b></i> denotes the vector transpose. <i>f</i> is usually chosen as a bounded monotonic function such as sigmoid. When 
<i>f</i> is a discontinuous step function, the nodes are often called <i>linear threshold units</i>.

The locus of points <b>x</b> with a constant sum SUM <i><sub>j</sub></i> ( <i>w<sub>j</sub> x<sub>j</sub></i> ) defines a 
hyperplane perpendicular to the vector <b>w</b>.

The graphic below shows 16 randomized single-layer networks each with different weight vectors shown in blue. The decision
boundaries that these weight vectors form is shown in red. Notice how the decision boundaries are always perpendicular
to the weight vector, since these decision boundaries are the locus of points <b>x</b> with constant sum of 0.


![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.1%20Hyperplane%20Geometry/images/1.png)


It is important to note that the orientation of the node hyperplane is determined by the direction of <b>w</b>. It does not depend
on the magnitude of <b>w</b>. The graphic below demonstrates how the angle of the hyperplane is not changed when only the magnitude
of <b>w</b> changes.


![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.1%20Hyperplane%20Geometry/images/2.png)


In order to obtain a decision boundary that does not pass through the origin, we can introduce a bias term.
Inclusion of a <i>threshold</i>, or <i>bias</i>, term <i>Θ</i>

<i>u</i> = <b>w</b><sup>T</sup><b>x</b> - <i>Θ</i>

shifts the hyperplane along <b>w</b> to a distance <i>d</i> = <i>Θ</i> / ||<b>w</b>|| from the origin. The graphic below shows how
the bias term changes the hyperplane.


![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.1%20Hyperplane%20Geometry/images/3.png)


The node nonlinearity <i>f</i> controls how the output varies as the distance from <b>x</b> to the node hyperplane changes.

When <i>f</i> is a binary hard-limitin function as in a linear threshold unit, the node divides the input space with a hyperplane,
producing 0 for inputs on one side of the plane and 1 for inputs on the other side.

With a softer nonlinearity such as sigmoid, the magnitude of <b>w</b> plays the role of a scaling parameter that can be varied
to obtain transitions of varying steepness. For large ||<b>w</b>||, the slope is steep and the sigmoid approximates a step
function. For small ||<b>w</b>||, the slope is small and <i>y</i>(<b>x</b>) is nearly linear over a wide range of inputs.

The graphic below shows how gradation changes when using a linear threshold unit, a sigmoid function with a small ||<b>w</b>||,
and a sigmoid function with a large ||<b>w</b>||.


![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.1%20Hyperplane%20Geometry/images/4.png)
