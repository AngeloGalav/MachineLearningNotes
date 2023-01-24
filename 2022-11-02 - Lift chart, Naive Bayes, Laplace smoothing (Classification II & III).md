# Classifier II - From Probabilistic to CRISP

## Lift chart
The lift chart can be used to extract CRISP values from a probabilistic result. Essentially, it can help us evaluate and _extract_ the examples that are _the most promising_ from a dataset. 

It is obtained by _sorting_ all the examples in order of _decreasing probability of being true_, if cons me can forecast how many of the true values we can find. 
It measures the capabilities how able is our classifier to "put data in the beginning of the chart" the example that are positive. So, ==the more curve we have in the beginning, the more effective is our classifier==. 

In short, the __lift chart__ essentially <u>tell us how able is our model to correctly classify objects as positive</u>. 
We have two lines:
- The _straight line_ plots the _number of positives_ obtained with a _random choice_ of a sample of test data.
- The _curve_ plots the _number of positives_ obtained drawing a fraction of test data with decreasing probability. 
![[lift_chat_example.png]]
The larger the area between the two lines, the better the classification model (==The more the curvature is high on the left, the better is our classifier==). 

The lift chart is especially useful when we have to work in batches. 

## ROC Curve
Keep in mind: Soft classifier = probabilistic classifier. 

Let's consider that we have a lot of data, and it has some noise. Some of this noise alters our data, and we have to distinguish the data to extract knowledge from it. In particular, we have to distinguish between hit rate and false alarm in a noisy channel, and so the noise alters the two levels according to a _Gaussian distribution_.
We could impose a threshold to the values, but we would have some FP and FN.

With less noise, the two Gaussian curves are better separated. 
![[roc_gaussian.png]]

Moving the threshold towards right increases both the rate of true positives and false positives caught. 

(N.B.: The curve on the bottom right of the image is not a lift chart. The values on the 2 axis are different (Probability of a TP on the y, Probability of a FP on the x)).

What we could do is imposing multiple thresholds, using __threshold steps__. Varying the threshold the behavior of the classifier changes, by changing the ratio of TP and FP.

So, the soft classifier (= probabilistic classifier) can be converted into a crisp one by choosing thresholds, in a ROC curve _the quality of the classifier_ is summarized by the __Area Under Curve__ (AUC) (the bigger, the better). 
Here's how to choose the thresholds:
```
sort the test elements by decreasing positive probability 
set the threshold to the highest probability, set TP and FP to zero 
repeat 
	- update the number of TP and FP with probability from the threshold to 1 
	- draw a point in the curve 
	- move to next top probability of positive en
end repeat
```

Here's an example of a ROC curve:
![[roc_example.png]]
The best soft classifier should give the higher score to all the positives, so that the shape of the curve catches all the positives at the beginning (and looks like a rectangular triangle almost).

Here's an example of a perfect selection:
![[perfect_roc.png]]

Here's an example of a random selection:
![[random_roc.png]]

-------
# What is the best method for classification? (Classification III)

Spoiler: it depends on the data. 

## Naive Bayes Classifier
It is Naive, and based on the Bayes theorem:
$$
Pr(d_1 = v_1, d_2 = v_2 | c = c_x)= Pr(d_1 = v_1 | c = c_x) \cdot Pr(d_2 = v_2 | c = c_x)
$$
It considers the contribution of all the attributes together (unlike DT which considers one attribute at a time). 
We _assume_ that each attribute is __independent__ from each other (i.e. height and weight are not independent, but color of hair and weight is).

Here's an example.
Given this data:
![[example_data.png]]

We will have:
![[example_text.png]]
(N.B.: we refer to likelihood of yes instead of probability of yes since the values are not normalized, meaning that sum of the likelihood values is not 1). 

Each likelihood is calculated by using the probability of each possible event condition.

### Bayes theorem
This theorem states that, given a hypothesis $H$ and an evidence $E$ that bears on that hypothesis:
$$
Pr(H|E) = \dfrac{Pr(E|H) \ Pr(H)}{Pr(E)}
$$
(In the previous example, the $H$ would be $play?$, while the $E$ would be all the values of the attributes). 
Usually, the ==hypothesis is the class==, say $c$, the evidence is the tuple of values of the element to be classified. 

We can split the evidence into pieces, one per attribute, and, if the attributes are independent inside each class, we can state that:
$$
Pr(c|E) = \dfrac{Pr(E_1|c) \ Pr(E_2|c) \ Pr(E_3|c)  \ Pr(c) }{Pr(E)}
$$
We can see $E$ as a _joint event_, where $E_1... E_3$ are (some) of the attributes that compose the event. 

So: 
- we compute the conditional probabilities from examples 
- Apply the theorem

Notice that the denominator is the same for all the classes, and is eliminated by the normalization step.

The Naive Bayes method is called _naive_ since the assumption of independence between attributes is quite simplistic. Nevertheless, ==it works quite well in many cases==.

##### Drawbacks
What if value $v$ of attribute $d$ never appears in the elements of class $c$?
In this case we have that $Pr(d = v | c) = 0$ . ==This makes the probability of the class for that evidence drop to zero==, and we would need to overcast it, which we don't want. 

We need an alternative solution. 

## Laplace smoothing
The basic idea of this other classifier is to _smooth_ the probability. 
Let’s start ignoring the details of the dataset, we consider only the value domains, and we know that for a given attribute $d$ there are $V_d$ distinct values. 

Then a simple guess for the frequency of each distinct value of $d$ in each class is $1/V_d$. 

We can smooth the computation of the posterior probabilities of values inside a class balancing it with the prior probability.

The parameters are:
- $\alpha$ , the _smoothing_ parameter, typical value is 1.
- $af_{d=vi,c}$  , the _absolute frequency_ of value $v_i$ in attribute $d$ over class $c$.
- $V_d$ , the _number of distinct values in attribute_ $d$ over the dataset.
- $af_c$ : _absolute frequency_ of class $c$ in the dataset.

We could also consider:
- the a-priori frequency, in which we don't consider the class.
- the a-posteriori frequency in which we consider the class. 

The final __smoothed frequency__ is:
$$
sf_{d=v_i, c} = \dfrac{af_{d=v_i, c} + \alpha}{af_c + \alpha V_d}
$$
- If $\alpha = 0$ , we obtain the standard, _unsmoothed_ formula (of the Bayes classifier).
- If $\alpha \rightarrow \infty$, then we would have that the frequency is $1/V$, in which we don't consider the class at all. In this case we consider the frequency of each value as equal, without any reference to a class.  
- if $af_{d=v_i, c} = 0$, then the $sf_{d=v_i, c}$ considers only the frequency of the class (and thus, takes in account, in someway, the different values of $d$). 

==The bigger is $\alpha$, the more importance we give to the prior probabilities== (without considering the class). 

In this case, ==the missing values do not affect the model==, and so it is not necessary to discard an instance with missing values (i.e., for decision trees, we cannot deal with missing values. In Naive-Bayes, if the value is missing (so, not 0!) we simply disregard the attribute). 

Normally, we simply discard that attribute with the missing values (in both train and test instances), as if it was not present. The descriptive statistics are based on Known, non-null values. 
- In test instances, it results in the likelihood will be higher for all the classes, but this is compensated by the normalization. 

#### Numeric values
With discrete values, frequencies are easy to count, and it is not so easy with real values. 
Since the method is probabilistic, we assume a _Gaussian distribution of the values_. 
From our data, we can easily compute the mean ($\micro$) and the variance ($\sigma$) inside each class. 
![[gaussian_distribution.png]]

- Probability and probability density are closely related, but are not the same thing. 
	- On a continuous domain, the probability of a variable assuming exactly a single real value is zero
	- A single value in the density function is the probability that the ==variable lies in a small interval around that value==.  
	- The value we use are, of course, rounded at some precision factor, and since that precision factor is the same for all the classes, then we can disregard it 
- If numeric values are missing, mean and standard deviation are based only on the values that are present

While these solutions provide excellent results in many cases, we would have a dramatic degradation if the simplistic conditions are not met. 
- Violation of independence – for instance, if an attribute is simply a copy of another (or a linear transformation), ==the weight of that particular feature is enforced== (something like squaring the probability)
- Violation of Gaussian distribution – use the standard probability estimation for the appropriate distribution, if known, or use estimation procedures, such as Kernel Density Estimation

## Linear classification with Perceptron
![[perceptron.png]]
A __perceptron__ is also called an _artificial neuron_. In practice, it's a linear combination of weighted inputs. 
The values in the boxes $d_1...d_D$ represents attributes, while the added value of $1$ with $w_0$ associated to it is the bias. 
The output is the weighted sum of these inputs. 

For a dataset with numeric attributes, we need to find (or learn) an _hyperplane_ (a straight line) such that all positives lay on one side and all the negatives on the other, and are separated.  
![[hyper_plane.png]]
As we know from [[2022-10-21 - Classification I#Example of a decision function|classification]], the task is: how can we learn this hyperplane from data?

#### The hyperplane formula
The hyperplane is described by a set of weights $w_0, ..., w_D$ in a linear equation on the data attributes $x_0, \dots, x_D$. 
The fictitious attribute $x_0 = 1$ is added to allow a hyperplane that does not pass through the origin (and it is like the bias of a perceptron). 
There are either none or infinite such hyperplanes. Let's assume there's a solution in this case though:
![[hyperplane_equation.png]]

##### Learning the hyperplane
```JS
set all weights to zero
	while there are examples incorrectly classified do
		for each training instance x do
			if x is incorrectly classified then // we need to change weights
				if class of x is positive then//this if is the core of the method
					add the x data vector to the vector of weights // weight ch. 
				else
					subtract the x data vector from the vector of weights
```
How can we prove that this "sum" in the weights is an actual improvement over the prior set of weights?

#### Linear perceptron convergence
Each change of weights ==_moves the hyperplane towards the misclassified instance_==, consider the equation after the weight change for a positive instance x which was classified as negative:
$$
(x_0 + w_0)*x_0, ..., (x_D + w_D)*x_D
$$The result is increase by a positive amount: 
$$
x_0^2 +  \dots + x_D^2
$$

Therefore, ==the result will be less negative or, possibly, even positive==. It is analogous for a negative instance which was classified as positive (but _we subtract_!).

The corrections are incremental and can interfere with previous updates.
==The algorithm converges if the dataset is __linearly separable__==, otherwise it does not terminate. For practical applicability, it is necessary to set an upper bound to the iterations.

Thanks to this method, we find one of the infinite possible solution. 

## Support vector machine (SVM)
It's a mathematical machine that tries to solve the problem of non-separability (no hyperplane that can separate the data).

A linear perceptron is inherently binary, since I can choose only one hyperplane (Naive-Bayes can deal with any number of values, so as we know it is not binary. This can also be seen in [[#Bayes theorem|this formula with the joint event]]). ==SVMs are also binary==.

How to overcome the linear separability? We could simply give up on the linearity, by using a different shape outside of the hyperplane. i.e.  ![[Screenshot_20221102_135147.png]]

This method would have some drawbacks:
- The method would become soon ==intractable for any reasonable number of variables==. 
	- i.e.,  with 10 variables and limiting to factors with maximum order 5 we would need something like 2000 coefficient 
- The method would be ==extremely prone to [[2022-10-26 - Classification II#Overfitting|overfitting]]==, if the number of parameters approaches the number of examples.

#### SVM key ideas
New efficient separability of non–linear functions that use __kernel functions__ to find the optimal parameter for separability, and so the problem of prediction shifted to a _function estimation_ problem.
- It is not a greedy search, in which at each step we move towards the most promising solution (unlike DT, which are greedy). 

The other idea is related to the following:
- When we deal with hyperplanes, we have an infinite amount of solution:
![[hyperplane_inf.png]]
- But what can we do to find the best one? In that set, the best line is this one ($B_1$). 
![[hyperplane_best.png]]
It is better because of the notion of __margin__. 
The __margin__ is ==defined as the distance between the lines that touch the nearest examples==. $B_1$ is the line with the biggest margin. 

If the margin of an hyperplane is small, it is more likely that an individual could fall into the ==wrong side==. 

The bigger is the margin, the safer is the behavior in respect of the new data that is similar to the older one. In this respect, it is essential to find the separation with the highest margin. 

Finding the support vectors and the maximum margin hyperplane belongs to the well known class of constrained quadratic optimization problems. 

#### Soft margin
If the margin is 0, the first step is to find an hyperplane which almost separated the classes (or we disregard the examples which generate a very narrow margin). 
![[narrow_margin.png]]

This way:
- Greater _robustness_ to individual observations.
- _Better classification_ of _most_ of the training observations.

A soft margin is obtained by adding a constraint to the optimization problem expressed by a single numeric parameter, usually called $C$. In this case, our $C$ is the _penalty_ parameter of the error term, and _controls the amount of overfitting_. Essentially, $C$ controls how much we are tolerant of bad classifications. 