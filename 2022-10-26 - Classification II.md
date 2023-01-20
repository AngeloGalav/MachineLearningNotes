# Training Set Error
A __training set error__ means that either my model based on the decision tree or the data are not able to discriminate completely the population.
This is because the model I generated with the available training data ==is not able to reproduce completely the ground truth==. 

To determine the training set error, we must execute the generated decision tree on the training set itself, and ==count the number of discordances between the true and the predicted class==: this is _the training set error_. The main causes of this are:
- the limits of decision trees in general: a decision tree based on tests on attribute values can fail 
- insufficient information in the predicting attribute

So, we have a limit in the type of the model (that is, the decision tree) or in the data. 

In our Iris dataset example, we had a training error of 1.35% (1 of 75 examples in the training set is not correctly classified by the learned decision tree). 
It is the error we make on the data we used to generate the classification model, and _it is the lower limit of the error_ that we make _with the training data_, which are the information we already know.

But, we are much more interested to an _upper limit_, or to a more significant value. Essentially, we are more interested in the error in the worst case. 

## Test Set Error
The __test set error__ is _more indicative_ of the expected behaviour with new data. Additional statistic reasoning can be used to infer _error bounds_ given the test set error.
![[test_set_error.png]]
As we can see, the test error is _much_ more than the training set error. This is due to something called overfitting. 

## Overfitting
Overfitting means, literally, that the model is too much fitted. 
Definition: overfitting happens when _the learning is affected by noise_. 
When a learning algorithm is affected by noise, the performance on the test set is (much) worse than that on the training set.

A decision tree is a _hypothesis_ of the relationship between the predictor attributes and the class. Here are some definitions: 
- $h$ = hypothesis
- $error_{train}(h)$ = error of the hypothesis on the training set 
- $error_x(h)$ = error of the hypothesis on the entire dataset

$h$ __overfits__ the training set if there is an alternative hypothesis $h'$ such that
![[overfitting_formula.png]]

What are the possible causes of overfitting?
1. Presence of noise 
	- individuals in the test set can have _bad values_ in the predicting attributes and/or in the class label.
2. Lack of representative instances 
	- Some situations of the real world can be _underrepresented_, or not represented at all, in the _training set_. 
	- This is quite common.

__Generalization__ the process is using the model fitted for a small portion of data with a bigger portion of the data, different from the previous. 

A good hypothesis has a low generalization error, which means it works well on examples different from those in training. 

### Occam's Razor theory
One of the thing we could do to reduce overfitting is to prune the decision tree. 
"Everything should be made as simple as possible, but not simpler"

- This means that simple theories are preferable to complex ones, and a long hypothesis that fits the data is more likely to be a coincidence. 
- _Pruning_ a decision tree is a way to simplify it, but we need precise guidelines for effective pruning. 

## General effect of model simplification
This is the effects of pruning summarized in a picture:
![[general_effect_model_simplification.png]]

What is in the middle of this graph is basically the _best situation_ we could possibly have, that is a balance between test set error and training set error.  

The situation shown in the figure happens with __ALL classification methods__, _not just decision trees_. This means that there's no best model, and we must try to find the optimal configuration. 

So, our model must have like a slider that we must use to tune the behavior of the fitting.  
In decision tree, we can tune the depth of the tree for example. 

For example, let's consider the case in which both nodes in a decision tree have the same information gain. The fitting process needs to make a choice between the 2 variables, and Scikit learn chooses randomly. 

This can be seen when using the code:
```python
random.state = 3
model = DecisionTreeClassificafier(random.state = random.state)
model.fit(x, y, max_depth = 2)
```
As we've said, _depth_, for a decision tree, basically defines the amount of pruning/simplification in the model. 

Let's suppose that we chose `random.state = 5`, and we instead have a training error of 3% instead of 5%. 
A good model should not give us surprises, so I should be able to forecast a worst case ever, like if an error shouldn't more than X%, so if I observe that changing only the random effects, the error changes from 3% to 5%, then I should not rely on 3%. 

What we could do is try some different random states randomly and consider the worst case. 

N.B.: `max_depth` parameter is an example of the $\theta$ we've talked about in the [[2022-10-21 - Classification I#Classificiation Model|previous lesson]]. These "sliders" are called __hyper-parameters__, and they influence how our model is generated. These parameters are chosen by the ML expert and must be optimized in order to achieve high precision. 

# Choice of the attribute to split the dataset
We are looking for the split generating the maximum __purity__. We need a measure for the purity of a node:
- a node with two classes in the same proportion has low purity 
- a node with only one class has highest purity

A node with high purity means that it has a low diversity of labels, so a node with only one class has high purity. 

### Impurity functions (measuring the impurity of a node)
- [[2022-10-21 - Classification I#Entropy|Entropy]] ([[2022-10-21 - Classification I#Information gain for binary split|Information Gain]] instead guides us to chose the best split and measures the change of purity).  
- __Gini Index__ (used by Scikit learn by default).  
- Misclassification Error (we skipped it).

#### Gini Index
Consider $p$ node with $C_p$ classes. Which is the frequency of the wrong classification in class $j$ given by a random assignment based only on the class frequencies in the current node?

For class $j$:
- frequency $f_{p,j}$
- frequency of the other classes $1- f_{p,j}$
- probability of wrong assignment $f_{p,j} * (1 - f_{p,j})$

The Gini index is __the total probability of wrong classification__:
$$
\sum_j f_{p,j} * (1 - f_{p,j}) = \sum_j f_{p,j} - \sum_j f^2_{p,j}
= 1 - \sum_j f^2_{p,j} 
$$

The maximum value is when all the records are uniformly distributed over all the classes: 
$1 - 1/C_p$.
The minimum value is when all the records belong to the same class: 0.

- When a node $p$ is split into $ds$ descendants, say $p_1, \dots, p_{ds}$. 
- Let $N_{p,i}$ and $N_p$ be the number of records in the $i$-th descendant node and in the root respectively.
- We choose the split giving the maximum reduction of the Gini Index:
![[gini_split_formula.png]]

### Algorithm for building DTs
There are several variants of DT building algorithms.
- We could also have tests based on linear combinations of numeric attributes, but these situation are very much complex. 

### Complexity of DT building
- N instances and D attributes in X 
	- tree height is $O(log N)$
- Each level of the tree requires the consideration of all the dataset (considering all the nodes) 
- Each node requires the consideration of all the attributes overall cost is $O(DN log N)$.

The fitting (pruning) requires to consider globally all instances at each level, generating an additional $O(N log N)$, which does not increase complexity (and so it is quite cheap). 

## Characteristics of DT Induction 
- It is a _non–parametric approach_ to build classification models. It does not require any assumption on the probability distributions of classes and attribute values.
	- "In statitics, the data is represented using a probability function, and I must find the parameters of this probability function" i.e. approximanting the distribution with a function (like Gaussian). 
- In principle, finding the best DT is NP-complete, since we are following a greedy approach.  Which means that the heuristic algorithms allow to find sub-optimal solution, but it is fast. 
	- It is suboptimal because we do not find the best DT (we will need to consider every DT possible!)
- Runtime cost of using a DT to classify new instances is extremely low ($O(h)$ where $h$ is the height of the tree). 
- ==Redundant attributes do not cause any difficulty==. 
	- An attribute is redundant if it has the same ability as another attribute to guess the label. 
- The nodes at a high depth are easily irrelevant (and therefore pruned), due to the low number of training records they cover 
- In practice, the impurity measure has low impact on the final result 
- In practice, the pruning strategy has high impact on the final result

N.B.: DTs are able to predict discrete values (the class) on the basis of _continuous_ or discrete predictor attributes. This is because the algorithm doesn't know that, for example, only integers values are allowed. 

---- 
# Model Selection
When we evaluate a model, we must consider a number of properties, and accuracy isn't necessarily the most important. There might be other types of errors or properties which do not influence accuracy, but are still important to consider.

i.e. if I have a model that is used to classify oil spills, which are a very rare occurance, if our model always says that there aren't any oil spills (even if they are), our model accuracy will still be 99%. 

## Traning set 
There's still the problem of _hyperparameter_ tuning. 
As [[#Overfitting|we've seen]], in supervised learning, the training set performance is __overoptimistic__.

- Evaluate how much the theory fits the data
- Evaluate the cost generated by prediction errors

- Empirically (and intuitively) the more training data we use the best performance we should expect 
- Statistically, we should expect a larger covering of the situation that can occur when classifying new data 
- We must consider the effect of random changes 
- The evaluation is independent from the algorithm used to generate the mode