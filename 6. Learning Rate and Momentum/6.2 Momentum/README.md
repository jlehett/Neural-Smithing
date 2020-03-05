## 6.2 Momentum

Back-propagation with momentum can be viewed as gradient descent with smoothing. The idea is to stabilize the weight trajectory
by making the weight change a combination of the gradient-decreasing term plus a fraction of the previous weight change.

Briefly momentum has the following effects:

* It smooths weight changes by filtering out high frequency variations. When the learning rate is too high, momentum tends to
suppress cross-stitching because consecutive opposing weight changes tend to cancel. 
* When a long sequence of weight changes are all in the same direction, momentum tends to amplify the effective learning rate to 
<i>η'</i> = <i>η</i> / (1 - <i>α</i>), leading to faster convergence.
* Momentum may sometimes help the system escape small local minima by giving the state vector enough inertia to coast over small
bumps in the error surface.

With momentum, the state vector has a tendency to keep moving in the same direction. Weight changes are affected by error
information from many past cycles—the larger the momentum, the stronger the lingering influence of previous changes. In effect,
momentum gives the weight state inertia, allowing it to coast over flat spots and perhaps out of small local minima.

A little inertia is useful for stabilization but too much may make the system sluggish; it may overshoot good minima or be unable
to follow a curved valley in the error surface. The system may coast past minima and out onto high plateaus where it becomes stuck.

The following graphs show <i>E</i>(<i>t</i>) curves for various momentum values and a fixed learning rate of 10.0, for networks
trained on the 4-bit parity problem.

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/6.%20Learning%20Rate%20and%20Momentum/6.2%20Momentum/images/1.png)

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/6.%20Learning%20Rate%20and%20Momentum/6.2%20Momentum/images/2.png)

At low momentum values, the <i>E</i>(<i>t</i>) curves are smooth and larger values of <i>α</i> lead to faster convergence. Occasional
spikes may occur but the system recovers quickly. As <i>α</i> increases past a certain point, however, convergence becomes
unreliable.
