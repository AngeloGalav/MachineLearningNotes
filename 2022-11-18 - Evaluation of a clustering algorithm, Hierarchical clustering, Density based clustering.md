# Evaluation of a clustering scheme
The evaluation of a _clustering scheme_ is related only to the _result_, not to the _clustering technique_. 

- Clustering is a non supervised method 
	- the evaluation is critical, because there is very little a-priori information, such as class labels. 
	- we need one or more _score (or index) function_ to measure various properties of the clusters and of the _clustering scheme_ as a whole
	- if some _supervised data are available_, they can be used to _evaluate the clustering scheme_.
- In 2D the clusters can be examined visually. In higher order spaces the 2D projections can help, but in general it is better to use more formal methods

### Issues on clustering evaluation
- Distinguish patterns from random apparent regularities
- Find the best number of clusters 
- _Non supervised_ evaluation 
- _Supervised_ evaluation 
- Relative comparison of clustering schemes

# Unsupervised evaluation
## Measurement criteria
- __Cohesion__ 
	- _proximity of objects in the same cluster should be high_ (they should be near) 
- __Separation__ between two clusters. We can measure it in several ways: 
		- _distance_ between the _nearest objects_ in the two clusters 
		- _distance_ between the _most distant objects_ in the two clusters 
		- _distance_ between the _centroids_ of the two clusters
These 3 separation types can be seen here:
![[this_image_cluster.png]]
- __Similarity – Proximity__ 
	- a two variable function measuring how much _two objects are similar_, _according to the values of their properties_.
	- Seen a [[2022-11-11 -  Preprocessing and dissimilarities#Similarity and dissimilarity (proximity functions)|ton of times]] [[2022-11-16 - Last remarks on Transformations, Introduction to Clustering (K-Means)#The proximity function|already]]. 
- __Dissimilarity__ 
	- a two variable function measuring _how much two objects are different_, according to the values of their properties (e.g. the Euclidean distance).
	- Seen also [[2022-11-11 -  Preprocessing and dissimilarities#Similarity and dissimilarity (proximity functions)|here]].

#### Cohesion
The __cohesion__ is the _sum_ of the _proximities_ between _the elements of the cluster_ and the geometric center (the _centroid_ or _medoid_). 
![[cohesion.png]]

- As we know, a centroid is a point in the space whose coordinates are the means of the dataset 
- A __medoid__ is an element of the dataset _whose average dissimilarity with all the elements of the cluster is minimal_. It is not necessarily unique, used in contexts where the mean is not defined, e.g. 3D trajectories or gene expressions. 
- While a centroid is not necessarily a part of a cluster, a medoid is. 
	- Association between average and median. 

#### Separation 
Separation between two clusters is measured through the proximity between the prototypes. 

#### Global separation of a clustering scheme
Can be computed by calculating the __Sum of Squares Between clusters__ (SSB). 
 $$
SSB = \sum^K_{i=1} N_i Dist(c_i, c)^2
$$
Where $c$ is the global centroid of the dataset. 

###### Total Sum of Squares 
The TSS is a global property of the dataset, independent from the clustering scheme. 
$$TSS = SSE + SSB$$

- $SSE$ is the [[2022-11-16 - Last remarks on Transformations, Introduction to Clustering (K-Means)#Sum of squared errors|Sum of Squared Errors]]. 
- $SSB$ is the [[#Global separation of a clustering scheme|Sum of Squares Between clusters]]. 

- for a given dataset, minimize SSE <=> maximize SSB

#### Clustering
In principle, we should consider all the possible clusterings of each number, so it is a huge number. 
What we have to do is find a shortcut.

## Evaluation of specific clusters and objects 
- Each _cluster can have its own evaluation_, and the worst clusters can be considered for additional split. 
- A weakly separated pair of clusters could be considered for merging 
- _Single objects_ can give _negative contribution_ to the _cohesion_ of a cluster or to the separation between two clusters (_border objects_).

### Silhouette score of a cluster 
Requirements for a clustering quality score 
- values are in a _standard range_, e.g. -1, 1 
- _increases with the separation_ between clusters. 
- _decreases for clusters with low cohesion_, or, in other words, with _high sparsity_.

The silhouette score is an _index_, meaning that _it does not have a measure_. 

Consider the individual contribution of each object, say $x_i$. 
We can compute the contribution of the individual to the cluster _sparsity_. 
![[sparsity.png]]
and the contribution to _separation_ from _other clusters_:
![[separation_contribution.png]]
![[Screenshot_20230125_163406.png]]

In this graph: 
- The average of the red distances is $a_i$
- The minimum of the two averages of green and blue distances is $b_i$

Given these definitions, the __Silhouette score__ of $x_i$ is:
$$
s_i = \dfrac{b_i - a_i}{\max(a_i, b_i)} \in [-1, 1] 
$$
- For the global score of a cluster/clustering scheme _compute the average score_ over the cluster/dataset
- If the score is negative, it means that it's a bad clustering (since $b_i < a_i$). In particular:
	- it means that there is a dominance of _objects in other clusters_ at a _smaller distance_ than _objects of the same cluster_. 

### Inertia and silhouette scores
![[inertia_and_silhoutte_scores.png]]
- Blue line is the inertia, while the orange line is the silhouette score. 
- The inertia improvement is a lot until the number of cluster reaches 10, and then increasing the number of classes doesn't result in a lot of improvement. 
	- Similarly, the silhouette score starts improving at 3 cluster, until it reaches 10 cluster, in which it declines.
We can see also in the same dataset how Silhouette scores change. 
![[silhoutte_scores.png]]

### Looking for the best number of clusters 
- Some algorithms, such as K-means, require the number of clusters as a parameters. 
- Measures, such as _SSE_ and _Silhouette_, are obviously influenced by the number of clusters, thus they can be _used to optimize K_. But... 
	- Computation of Silhouette score _is expensive_!
	- SSE decreases monotonically for increasing K. 

#### Elbow method
For this reason, we must consider _inertia_ as a possible parameter for deciding $K$.  
The _inertia_ varying $K$ has frequently _one or more points where the slope decreases_ (i.e. 10 clusters in the previous image): one of this points is frequently _a plausible value for $K$_.
This is called __elbow method__. 

The _silhouette score_ varying $K$ has frequently _a maximum_, in this case it indicates the best value for $K$ \[so, even if it is very expensive, it can be used for small datasets\].

# Supervised evaluation

## Gold standard
Consider a partition of a dataset _similar_ to the data _to be clustered_, which we call __gold standard__, and defined by a _labelling scheme_ $y_g(.)$ . 
- it is the same as the _labels attached to supervised data_ for training a classifier.
- In clustering, using the gold standard is the same as doing training and test on two different datasets. 
	- Meaning that I can cluster ignoring the labels first, and then recluster the same data with the labels and compare the results.  
	- If there's a good agreement between the 2 methods, then we can use the model that we've used for clustering with other, unsupervised data. 

Now, consider a _clustering scheme_ $y_k(.)$. 
- the _cardinalities of the sets of __distinct__ labels_ generated by the two schemes $V_g$ and $V_k$ _can be different_, and also in case of identity of the two grouping schemes, a permutation of labels could be necessary to make them equal. 

The gold standard technique can be used to _validate a clustering technique_, which can be applied later to new, unlabeled data.
- the purpose is quite similar to testing a classifier. The difference is that in this case _we are more interested in grouping new data_ than in labelling them following the Gold Standard scheme.

In practice, comparing a cluster with a gold standard is similar to testing a classifier. 
![[cluster_example.png]]

What is shown by this image is the confusion matrix obtained through the gold standard procedure. Strangely, the large values are not on the diagonal (like in normal confusion matrices). This is because the labels are assigned randomly by the gold standard, and are essentially different from the original values of the labels. 
- but, if we remap the values obtained, we will obtain a standard confusion matrix.  

#### Similarity oriented measures
Consider again a _gold standard_ $y_g(.)$ and _clustering scheme_ $y_k(.)$. 
Any pair of objects can be labelled in 4 different ways:
![[agdk_culo.png]]

In a optimal situation (the perfect assignment), the number of elements labelled SGDK or DGSK should be 0. 

# Hierarchical clustering 
Another clustering methods. It generates a nested structure of clusters. It can be of two types:
- __Agglomerative (bottom up)__: as a starting state, _each data point is a cluster_. 
	- in each step _the two less separated clusters are merged into one_
	- a measure of separation between clusters is needed 
- __Divisive (top down)__: as a starting state, the _entire dataset is the only cluster_. 
	- in each step, _the cluster with the lowest cohesion is split_
	- a measure of cluster cohesion and a split procedure are needed

We won't see the divisive approach. 

#### Output of hierarchical clustering
![[output_hierchical.png]]
- Dendrogram (left) 
- Nested cluster diagram (right) 
- They represent the same structure 
- The representation is the same for agglomerative and divisive.

## Separation techniques in hierarchical clustering
![[separations_in_hierarchical.png]]

- In _single link_, we measure that the separation between the two clusters as the _minimum distance_ between _pairs of objects_ in the two clusters.
- In _complete link_, we measure that the separation between the two clusters as the _maximum distance_ between _pairs of objects_ in the two clusters.
- In _average link_, we measure that the separation between the two clusters as the _average distance_ between _pairs of objects_ in the two clusters. 

#### Other separation techniques
We could use the distance between the centroids, or: 
- __Ward’s method__: given two sets with the respective [[2022-11-16 - Last remarks on Transformations, Introduction to Clustering (K-Means)#Sum of squared errors|SSE]], the separation between the two is measured as the _difference_ between the _total SSE_ resulting _in case of a merge_ and _the sum or the original SSEs_; 
	- _smaller separation_ implies a _lower increase_ in the SSE after _merging_. 
	- i.e. Considering two clusters 1 and 2: $SSE_{1,2} - (SSE_1 + SSE_2)$. 

There's a nice example of agglomerative hierarchical clustering in slides 78 [here](https://virtuale.unibo.it/pluginfile.php/1336023/mod_folder/content/0/machineLearning-04-clustering.pdf?forcedownload=1). 

## Single linkage algorithm
Algorithm for clustering based on agglomerative hierarchical clustering, using _single link_.

What we realize for this algorithm is that we need to compute _a lot of distances_.
So, we can compute a _distance matrix_, so that we can lookup the distances without computing them at each step.  

Here's the algorithm:
- While the number of clusters is greater than 1
	- find the _two clusters_ with the _lowest separation_, say $k_r$ and $k_s$ 
	- _merge them_ in a cluster 
	- _delete_ from the distance matrix _the rows_ and columns $r$ and $s$ and _insert one new row_ and _column_ with the _distances of the new cluster from the others_. 
	- The new distance is computed as:
![[formula_simple_linkage.png]]

#### Complexity of SLA
- Space and time: $O(N^2)$ for the computation and the storage of the _distance matrix_. 
- Worst case: $N - 1$ iterations to reach the final single cluster. 
- For the $i$-th step of the main iteration:
	- search of the pair to merge $O((N -i )^2)$
	- recomputation of the distance matrix $O(N - i)$ 
- Time, _in summary_: $O(N^3)$.  Can be reduced to $O(N^2 log(N))$ with indexing structures. 

There's a nice example of clustering Italian cities in slides 86 [here](https://virtuale.unibo.it/pluginfile.php/1336023/mod_folder/content/0/machineLearning-04-clustering.pdf?forcedownload=1), as well as another example using animals (and complete linkage!). 

# Density based clustering
It overcomes another problem of K-Means, namely the fact that it is not very good with [[2022-11-16 - Last remarks on Transformations, Introduction to Clustering (K-Means)#Pros and cons of K–means|convex datasets]] (hyperspherical clusters). 
![[strange_datasets.png]]
In density based clusterings, clusters are _high–density regions_ separated by _low–density regions_.

How can we define _density_? The two most obvious solutions are:
- __Grid–based__ 
	- split the (hyper)space into a _regularly spaced grid_ (in hypercubes essentially)
	- _count_ the number of objects _inside each grid element_ 
- __Object–centered__ 
	- _define the radius_ of a (hyper)sphere (a radius of interest). 
	- _attach to each object_ the _number of objects_ which are _inside that sphere_. 

Ok nice, but how do we actually define _the value_ for which a set of values is considered dense? Well, it depends on the dataset...

## DBSCAN – Density Based Spatial Clustering of Applications with Noise
Let's consider some data put in this way:
![[dbsn_example.png]]
In this set, intuitively, $p$ is a border point, while $q$ is a core point.
We define a ==radius $\epsilon$ ==and define as _neighborhood of a point_ the $\epsilon$–_hypersphere_ centered at that point.
Points $p$ and $q$ are one in the neighborhood of the other: neighborhood is symmetric.

We then define a ==threshold `minPoints`== and define as _core_ a point with _at least `minPoints` points in its neighborhood_, as _border_ otherwise.

#### Direct density reachability
We define that a point $p$ is __directly density reachable__ from point $q$ if:
- $q$ is core 
- $q$ is in the neighborhood of $p$

So, a core point can attract a border point that is in its neighborhood. And so, we want to build clusters starting from core points and collecting companions that are in the neighborhood. 
![[direct_reachability.png]]
- _direct density reachability is not symmetric_: in the example $q$ is not directly density reachable from $p$, since $p$ _is border_. 
	- a border point cannot reach anyone. 

#### Density reachability
A point $p$ is __density reachable__ from point $q$ iff:
- $q$ is core 
- there is a sequence of points point $q_i$ such that $q_{i+1}$ is directly density reachable from $q_i$ , $i \in [1, nq],$ $q_1$ is directly reachable from $q$ and $p$ is directly density reachable from $q_{nq}$. 
- in short: there's a sequence of directly reachable core points from $q$ to $p$.  

- reachability is not symmetric. 
![[density_reachability.png]]

#### Direct connection
- a point $p$ is _density connected_ to point $q$ iff there is a point $s$ such that $p$ and $q$ are density reachable from $s$. 
- density connection is symmetric
![[density_connection.png]]

#### Generation of clusters
In this context, a _cluster_ is a _maximal set of points_ connected by _density_.
_Border points_ which are _not connected_ by density _to any core_ point _are labelled as noise_.

Of course, if there are outliers, there might be a problem (connection between 2 clusters).

## How to set $\epsilon$ and `minPoints`?
- As in many other machine learning algorithms, a __grid search__ over _several combinations_ of _hyperparameters_ can be useful.
- As a rule of thumb, you can try $minPoints = 2 * D$, the number of dimensions.
- Noise suggest an increase in minPoints
- A guess for $\epsilon$ requires more effort, considering the distance of the k–nearest neighbor, with $k = minPoints$

- if I increase $\epsilon$, it means that I will look further, and some border points will become core points. 
	- we have this same situation if instead of increase $\epsilon$ I decrease `minPoints`. 
- Decreasing $\epsilon$ and increasing `minPoints` _reduces the cluster size_ and increases the number of _noise points_.

## Good guess for $\epsilon$ 
Consider the vector of the __k-distances__:
- choose k 
- for each point we compute _the distance of its k–nearest neighbor_, and we sort the points for _decreasing k–distance_. 

- Choosing a given k–distance as $\epsilon$, it turns out that all the points with a _k–distance bigger than $\epsilon$_ will be considered as _border_. 

Usually, datasets which exhibit some tendency to clustering exhibit also a change of slope. The best $\epsilon$ can be found with a _grid search_ in the _area of the change of slope_.

![[guess_epsi.png]]

### DBSCAN, PROS AND CONS
##### Pros
- Finds clusters of any shape 
- Is robust w.r.t. noise 
###### Cons
- Problems if clusters have widely varying densities

Complexity is $O(N^2)$, reduced to $O(N log(N))$ if we use spatial indexes. 

## Kernel Density Estimation (KDE)
- The overall density function is the sum of the _influence functions_ (or _kernel functions_) _associated with each point_.
	- The kernel function must be symmetric and monotonically decreasing
	- usually has a parameter to set the decreasing rate
- It's very good for vector spaces. 
![[kde.png]]

### DENCLUE algorithm
![[decnlue.png]]
\[not so important, the professor skimmed through it.\]

### DENCLUE comments
##### Pros
- It has a strong theoretical foundation on statistics precise computation of density. 
	- DBSCAN is a special case of DENCLUE where the influence is a step function  
- Good at dealing with noise and clusters of different shapes and sizes 
###### Cons
- expensive computation $O(N^2)$ (can be optimized with approximated grid based computation) 
- Troubles with high dimensional data and clusters with different densities


-----

This Lesson was a fucking pain to listen. It was very difficult to understand some of the meanings and the concepts were pretty complex.  