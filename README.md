# ft_hinton86
c++ implementation of http://www.cs.toronto.edu/~fritz/absps/ieee-lre.pdf
Originally, I uploaded this source code to my blog(sephiroce.com) in 2016 as a h/w for a Neural network lecture.

The purpose of this paper(Hinton '86) is teaching NN to understand relationships between P1 and P2.

<b>Family Tree</b>

![ht86_0](images/ht86_0.png)

The family tree above can be expressed by a set of tripletts.

Person1	Relation	Person2
Christopher	wife	Penelope
Arthur	wife	Margaret
James	wife	Victoria
Andrew	wife	Christine
Charles	wife	Jennifer
Roberto	wife	Maria
Pierro	wife	Francesca

The family tree contains 104 triplett.(FamilyTree.xlsx)

<b>Neural Network Structure</b>

![ht86_1](images/ht86_1.png)

<b>Feed forward</b>
There are not bias nodes.
<a href="http://www.codecogs.com/eqnedit.php?latex=^{x_{j}}=\sum&space;_i&space;y_iw_{ji}" target="_blank">
<img src="http://latex.codecogs.com/gif.latex?^{x_{j}}=\sum&space;_i&space;y_iw_{ji}" title="^{x_{j}}=\sum _i y_iw_{ji}" /></a>\r\n\r\nNon-linear function(sigmoid function)\r\n<a href="http://www.codecogs.com/eqnedit.php?latex=y_j=\frac{1}{1+e^{-x_j}}" target="_blank">
<img src="http://latex.codecogs.com/gif.latex?y_j=\frac{1}{1+e^{-x_j}}" title="y_j=\frac{1}{1+e^{-x_j}}" /></a>
The units are arrainged in layers with a layer of input uints at the bottom, any number of intermediate layers, and a layer of output uints at the top. There no feedback connections.

<b>Back propagation</b>
Squared residual errors, no bias node, Batch mode

acceleration medthod : delta W(t-1)

<a href="http://www.codecogs.com/eqnedit.php?latex=\Delta&space;w(t)=-\varepsilon&space;\frac{\partial&space;E}{\partial&space;w(t)}+\alpha&space;\Delta&space;w(t-1)" target="_blank">
<img src="http://latex.codecogs.com/gif.latex?\Delta&space;w(t)=-\varepsilon&space;\frac{\partial&space;E}{\partial&space;w(t)}+\alpha&space;\Delta&space;w(t-1)" title="\Delta w(t)=-\varepsilon \frac{\partial E}{\partial w(t)}+\alpha \Delta w(t-1)" /></a>

t is incremented by 1 for each sweep through the whole set of input-output cases, and alpha is an exponential decay factor between 0 and 1 that determines the relative contribution of the current gradient and ealier gradients on the weight change.

The results.
![ht86_2](images/ht86_2.png)
![ht86_3](images/ht86_3.png)

Weights from the 24 input units for people

![ht86_4](images/ht86_4.png)

Weights from the 12 input units for relations

![ht86_5](images/ht86_5.png)

My model got 2 out of 4 test cases,
wheree "correct" means that the output unit corresponding to the right answer had an activity level above 0.5, and all the other output units were below 0.5.
