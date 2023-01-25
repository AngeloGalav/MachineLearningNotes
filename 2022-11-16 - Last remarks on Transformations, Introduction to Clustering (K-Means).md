##### Continuation on transformations
Attribute transformations which change data distribution can be useful when data is skewed (meaning that there are large agglomerations of data, with some very evident outliers). 
![[skewed_data.png]]
We could use a `PowerTransformer` to make data less skewed. 
Here's an example of scaled data (standardization):
![[scaled_data.png]]
Here's what happens to distances before and after scaling:
![[scaling_distance.png]]
Before the scaling the two distances seemed to be very different, due to a big numeric difference in the `Salary` attribute, now _they are comparable_.

### Transformations operating on a _single_ attribute
- __Range–based scaling__ stretches/shrinks and translates the range, according _to the range of the feature_ (there are some variants) 
	- good when we know that the data are not Gaussian, or we do not make any assumption on the distribution 
	- the base variant, the __MinMax scaler__, remaps to \[0, 1\]
- __Standardization__ _subtracts the mean_ and _divides by the standard deviation_ ($\dfrac{x - \micro}{\sigma}$) 
	- the resulting distribution has mean zero and unitary standard deviation 
	- _good when the distribution is Gaussian_ 
	- `StandardScaler`

### Transformation in Scikit-Learn
These are affine transformations: linear transformation plus translation 
- `MinMaxScaler` – remaps the feature to \[0, 1\] 
- `RobustScaler` – centering and scaling statistics is based on _percentiles_
	- less influenced by outliers (in similar way to medians). 

\[the prof said this were important, so I added them\]

#### Normalization
Normalization is mentioned sometimes with different meanings, frequently it refers to `MinMaxScaler`. 
- in Scikit–learn the Normalizer normalizes _each data row_ to _unit norm_ (it considers all the attributes of an individual)
	- Data is mapped onto a circle (circumference to be exact) with the center on the origin and $radius = 1$. 

### Workflow in transformation 
1. transform the features as required both for the train and test data 
2. fit and optimize the model(s) 
3. test 
4. possibly, use the original data to plot relevant views (e.g. to plot cluster assignments)

## Unbalanced data in classification
- The _performance minority class_ (classes) has little impact on standard performance measures 
- The optimized model could be _less effective_ on _minority class_ (classes) 
- As we saw, some estimators allow [[2022-10-28 - Evaluation of performance of a classifier (Classification II)#Evaluation of a classifier|to weight classes]] 
- Some performance measures allow to take into account the contribution of minority class (classes)

### Cost Sensitive learning
- several classifiers have the parameter `class_weight`, which changes the cost function to take into account the unbalancing of classes
- in practice _it is equivalent to __oversampling__ the minority class_, (repeating random examples) in order to produce a balanced training set.

##### Undersampling
- Obtains _a balanced training set_ by randomly _reducing_ the number of examples of the _majority class_ 
- Obviously part of the knowledge embedded in the training set is dropped out

##### Oversampling with SMOTE (Synthetic Minority Oversampling Technique)
It synthesizes _new examples_ from _the minority class_, so it is a type of _data augmentation_.
Here's how it works:
- a random example from the minority class is first chosen 
- $k$ of the nearest neighbors are found 
- a _randomly selected neighbor_ is chosen, and a _synthetic example_ is created at a randomly _selected point between the two examples_ in feature space. 

This method is more robust to generalization, since it is essentially duplicating with some noise (by adding data that is somewhat plausible). 

------
# Clustering
- Given a set of $N$ objects $x_i$, each described by $D$ values $x_{id}$. 
- Our task is to _find a natural partitioning_ in $K$ __clusters__ and, possibly, a number of _noise objects_ 
- The result to solving a __clustering scheme__, i.e. a _function_ mapping each _data_ object to _the sequence_ $[1 . . .K]$ (or to noise) 

- Desired property of clusters: objects in the same cluster are similar 
	- look for a clustering scheme which maximizes intra–cluster similarity
  
In formal wording:
![[clustering_formal.png]]

So, clustering in a sense is similar to classification, but _we don't have a ground truth_, there's nothing to compare the data.
![[clustering_image.png]]
There are many ways that we can cluster data:
![[clustering_2.png]]
Which of the two is better, i.e. _maximizes intra–cluster similarity_? A measure is needed.

## Centroid
- A __centroid__ is a _point_ with coordinates computed as _the average of the coordinates_ of all the points in the cluster. 
- in physics, it is the center of gravity of a set of points of equal mass.
- for each cluster $k$ and dimension $d$, the $d$ coordinate of the centroid is defined as:
![[centroid_definition.png]]
Here's the centroid on the 2 previous clusters:
![[centroid_clusters_example.png]]

# K-means

## Understanding K-Means
\[This first part is only related to understanding the method. You can skip most of this if you already undestand\]

Let's say that we have some data, which is distributed like this. 
![[data_example.png]]
We can clearly see that there are 5 clouds of data. But can we compute that there are (effectively) 5 clouds in a D-dimensional space?

To understand, let's use the metaphor of a transmission of data. 
- We transmit the coordinates of points, in particular we want to transmit where the points are distributed in the space. 
- Allow only _two bits_ per point in _the transmission_, so it will be lossy 
	- We need a coding/decoding mechanism.
- The loss is equal to the _sum of the squared errors_ between the _real points_ and their _encoding/decoding_.
	- The coding/decoding mechanism must minimize this loss.

First idea for space representation:
- partition the space into a grid of cells 
- decode each pair of bits with the center of the grid cell.
![[first_idea.png]]
- to reduce the loss (and improve the strategy), we can take into account how the original data is distributed between the quadrants: we _decode each pair of bits_ with _the centroid_ of the points in the grid cell.
![[second_strategy.png]]
We can even do better though!
- Let's cheat a bit, and decide that we can decide the number of partitions (clusters). 
- We ask the user the number of cluster K 
- From the data, we select K randomly, as _temporary centers_. These are also our decoding points. 
![[temporary_centers_example.png]]

At this points, we can partition the points by using the [[2022-11-04  - SVMs, Neural Networks (Classification III)#K-nearest neighbors classifier (or KN).|k-nearest neighbors]] algorithm. 
- _Each point_ finds _his nearest center_ and is labelled (i.e. colored) accordingly. 
![[initial_labeling.png]]
Now that I have the coloring scheme, it is reasonable that each star is placed in a place that minimizes the loss (the centroid). 
- For each center, _finds the centroid_ of its points, and _move there the center_.
- ==Repeat the K-nearest neighbors classification==, and find the centroid again etc...

## Remaining questions to solve
1. [[#Question 1: Distortion|What are we trying to optimize?]] 
2. [[#Question 2: Algorithm termination|Is termination guaranteed?]] 
3. [[#Question 3: Local or global minimum?|Are we sure that the best clustering scheme is found]]? 
	- Which is the definition of best clustering scheme? 
4. [[#Question 4: Looking for a good ending state|How should we start?]]
5. [[#Question 5: Choose the number of clusters|How can we find the number of clusters?]]
	- Normally, we can't just ask the user for $K$...

### Question 1: Distortion
We are trying to optimize the _distortion_, frequently called in the literature/physics Inertia.
- The distortion is the sum of the _differences_ between _the point_ and _the centroid of its cluster_. 

![[distortion_definition.png]]
N.B.: in our case, the encoding is the coloring (cluster), the decoding is the centroid. 

The _distortion_ is the value that we want to minimize. 
Which properties are requested to $c_1, ..., c_K$ for the minimal distortion?
1. $x_i$ must be encoded with the nearest center. 
![[c_encode.png|center]]
(I want to find the value $j$ minimizing that value). 

2. The _partial derivative of distortion_ w.r.t. the position of each center _must be zero_, because in that case the function has either a maximum or a minimum.
![[distortion_derivative.png]]
![[distortion_minimal.png]]
Naturally, we obtain the formula of the centroid. All this because you wanted a stupid answer to a question you may not even ask.  

So, to achieve minimal distortion, we must perform these operations alternately: 
1. $x_i$ must be encoded with the nearest center 
2. each center must be the centroid of the points it owns

### Question 2: Algorithm termination
It can be proven that after a finite number of steps the system reaches a state where neither of the two operations changes the state. 

- There is only a _finite number of ways to partition_ $N$ objects into $K$ groups
- The state of the algorithm is given by the two encode/decode functions 
- The number of configurations where all the _centers are the centroids of the points_ they own is _finite_ 
- If after one iteration the state changes, _the distortion_ is _reduced_. 
- Therefore, each change of state bring _to a state which was never visited before_.
	- \[Asperti would say "we're in a simpler state than before, so convergence is ensured"\]
- In summary, ==sooner or later the algorithm will stop because there are no new states reachable==.

### Question 3: Local or global minimum?
Is the ending state the best possible? Not necessarily. An example:
![[local_global_minima.png]]
A random assignment of the centers places the center in two very bad spots. 
We would want one point in the cloud on the left and two points in the two clouds on the right, but our algorithm can't move from here. 
This is because the algorithm is of the _greedy_ type, and doesn't necessarily explore all the possible states. 
So, the algorithm guarantees only to find a _local minimum_, and is sub-optimal.

- We can try to relax this problem by inserting initial random centers that are not too close to each other. 

### Question 4: Looking for a good ending state
_The starting point is important_! 
- choose randomly the first starting point 
- choose in sequence the $2..K$ starting points _as far as possible_ from the preceding ones.
- Re-run the algorithm with different starting points

### Question 5: Choose the number of clusters
not so easy... 
- _try various values_ 
- use a quantitative evaluation of the quality of the clustering scheme to decide among the different values 
- the best value finds the optimal compromise between the minimization of intra-cluster distances and the maximization of the inter cluster distances.

In general, when we increase the number of clusters, the _distortion decreases_.   

## The proximity function
We've discussed about [[2022-11-11 -  Preprocessing and dissimilarities#Similarity and dissimilarity (proximity functions)|proximity functions in the past]]. What proximity function should we use?
- The most obvious solution, used in the previous formulas is the _Euclidean distance_. 
	- good choice, in general, for vector spaces
- Several alternative solutions for specific data types and data sets see the "Data" module for additional discussions.

## Sum of squared errors
The official name of the [[#Question 1: Distortion|distortion]] is the __Sum squared errors__ (SSE).
![[SSE.png]]
We've also already seen [[2022-11-09 - Ensemble methods, Regression#Quality of the fitting|SSE here]]. 

- _A cluster $j$ with high $SSE_j$ has low quality_. 
- $SSE_j = 0$ if and only if all the points are coincident with the centroid. 
- SSE decreases for increasing K, is zero when $K = N$ 
- => _minimizing SSE is not a viable solution to choose the best K_. 

#### Empty clusters
It may happen, at some step, that a centroid does not own any point. 
Thus, we may need to _choose a new centroid_.

#### Outliers
__Outliers__ are points _with high distance from their centroid_. 
They have high contribution to SSE, and have a bad influence on the clustering results. 
Sometimes _it is a good idea to remove them_, the choice is related to the application domain.

### Uses of K-means
- It can be easily used in the beginning, for the _exploration of data_.  
- In a one-dimension space, it is _a good way to [[2022-11-11 -  Preprocessing and dissimilarities#Discretization|discretize]] the values_ of a domain in non-uniform buckets. 
- Also, it is _very fast_.
- Used for choosing the color palettes, GIF compressed images: color quantization, vector quantization. 

### Complexity of K-mean
Complexity is $$O(TKND)$$where:
- $T$ number of iterations 
- $K$ number of clusters 
- $N$ number of data points 
- $D$ number of dimensions

## Pros and cons of K–means
#### Pros 
- Fairly efficient, nearly linear in the number of data points in general $T,K, D << N$
#### Cons
- in essence it is defined for spaces where the centroid can be computed 
	- e.g. when the Euclidean distance is available, also other distance functions work well 
	- _cannot work with nominal data_ 
- _requires the K parameter_ 
	- nevertheless the best K can be found with iterations 
- it is very _sensitive to outliers_ 
- does _not deal with noise_ 
- does _not deal properly with non-convex clusters_.
	- A cluster that has a concavity in this space. 