# Ensemble methods
Ensemble methods (aka Classifier combination) is based on the idea that instead of using a single classifier, we can use several classifiers that work together.
So, it is based on training a set of __base classifiers.__
The final prediction is obtained _taking the votes of the base classifiers_.
In general, ensemble methods tend to perform better than a single classifier.

### Why should this be better?
Let us consider 25 binary classifiers, each of which has error rate $e = 0.35$.

The ensemble classifier output is the majority of the predictions (13 or more). 
- If the base classifiers are equal, the ensemble error rate is still $e$ (so this operation is useless)
- If the base classifiers _are completely independent_ and have all error rate $e$, i.e. their errors are uncorrelated, then _the __ensemble__ will be wrong only when the __majority__ of the base classifier is wrong_.
$$
e_{ensemble} = \sum_{i=13}^{25} 
\begin{pmatrix}
25 \\ i
\end{pmatrix}
e^i(1-e)^{25-i} = 0.06
$$
So, we need to combine all the errors of the classifiers. And what we obtain is that the errors of the ensemble is much smaller (0.06) than the error of each classifier (0.35).  
\[Avengers, ensemble!!!\]

The catch is that _it doesn't it work with every error rate!_.
![[base_ensemble_comparison.png]]
If the base error rate is 0.5, the resulting error rate is still 0.5. So, it works only if the base error rate is $e \le 0.5$. 

In general, ensemble methods are useful if:
1. the base classifiers are independent
2. the performance of the base classifier is better than random choice (0.5)
	- If we're near random choice (i.e. $e = 0.45$) then it won't help, but if we're at $e = 0.35$ we'll get a good performance. 

## Creating independence
But how can we make classifiers so that they're independent? How can we make 2 completely different decision tree?

#### By manipulating the training set
With a _manipulation of the training_ set we can obtain _some independence_. 
There are 2 major methods for dataset manipulation:
- __Bagging__: we _resample the training set repeatedly_, with replacement (in the training set) according to a uniform probability distribution.
	- i.e. 3 different subsets of the training set to 3 classifiers. 
	- There's some independence, but not total independence. 
	- _Sampling with replacement_ means that a value is reinserted after it has been extracted. 
- __Boosting__: iteratively changes the distribution of training examples so ==that the base classifier focus on examples which are hard to classify==. 
	- I train my classifier and I predict the examples, some of them will be correctly predicted, the others won't.
	- I will increase the probability of being chosen for the bad predictions. In this way, I will force my classifier to focus on the examples that are badly predicted. 
- Adaboost: the importance of each base classifier depends on its error rate. 

Boosting and Bagging manipulate the training set, Adaboost manipulates the combination of the final results. 

#### By manipulating the input features
_Subset of input features_ can be chosen either random or according to suggestions from domain experts. Meaning, we use different _subset of attributes_ for each classifier. 
__Random forest__: uses _decision trees_ as _base classifiers_. It frequently produces very good results. 

This is effective only if we have a large number of features/attributes. 

#### By manipulating the class labels
When we have a large number of class labels, for each base classifier we randomly partition the class labels into 2 subsets, say A1, A2 and re-label the dataset.
We then train binary classifiers with respect to the two classes. 
At testing time, when a subset is selected, all the classes that it includes will receive a vote. 
As always, the class with the top score will win.

Here we have a general schema for the ensemble methods. 
![[general_schema_ensemble.png]]

### Forest of randomized trees
A diversified set of classifiers is created by introducing _randomness_ in the classifier construction. 
Each tree in the ensemble is _built from a sample drawn with replacement_ (i.e., a bootstrap sample) from the training set. 
Furthermore, when splitting each node during the construction of a tree, the best split is found either from all input features or a _random subset_ of size `max_features`.

#### Bias–vs–Variance tradeoff
- __Bias__ is the _simplifying assumptions_ made by the model to make the _target function easier to approximate_
	- If a model has a bias, usually it means that it is a very simplified model
- __Variance__ is the _amount that the estimate of the target function will change_, given different training data
- Bias-variance trade-off is the sweet spot where our machine model performs between the errors introduced by the bias and the variance

"If Bias vs Variance was the act of reading, it could be like Skimming a Text vs Memorizing a Text"

The purpose of the two sources of randomness is to _decrease the variance_ of the forest estimator. 
- ==individual decision trees typically exhibit high variance and tend to overfit== 
- the _injected randomness_ in forests generates decision trees with somewhat decoupled prediction errors.  
- by taking an average of those predictions, some errors can cancel out
- random forests achieve a reduced variance by combining diverse trees, sometimes at the cost of a slight increase in bias.
- In practice the variance reduction is often significant hence yielding an overall better model.

### Boosting
In some cases, we can give weights to classes, or _to each training instance_ (so that, for example, an error in a certain instance should be more penalizing). 
Different classifiers are trained with those weights, and _the weights are modified iteratively according to classifier performance_. 
- For example, I increase the weights of the individuals which are badly classified, so that it the next round the model will focus on them. 

The idea is "change the weights in order to compensate for the errors".  

#### AdaBoost
- fit a sequence of _weak learners_ on _repeatedly modified versions of the data._ 
- the _predictions_ from all of them are then _combined_ through a _weighted majority vote_ (or sum) to produce the final prediction. 
	- Majority vote: the prediction given by the hightest number of classifiers wins. 
- the data modifications at each so-called __boosting iteration__ consist of _applying and modifying weights_ to each of the training sample.

- initially, the weights $w_1, w_2, . . . , w_N$ are all set to $1/N$, so that the first step simply trains a weak learner on the original data. 
- for each successive iteration, the _sample weights_ are _individually modified_ and the learning algorithm is reapplied to the _reweighted data_. 
	- at a given iteration, ==the training examples that were incorrectly predicted by the boosted model induced at the previous step have _their weights increased_==, whereas the _weights are decreased_ for those that were _predicted correctly_. 

- as iterations proceed, _examples that are difficult to predict receive ever-increasing influence_; each subsequent weak learner is thereby forced to concentrate on the examples that are missed by the previous ones in the sequence.

---------
# Regression

__Regression__ is a _supervised_ activity, and the workflow for regression is similar to the workflow of classification. 
- The target variable is numeric 
- Our objective is to _minimize the error_ of the prediction with _respect to the target_. 
	- We can't just count the wrong predictions like in classification. 

While classification considers as a target a discrete value (i.e. an int), most preferably with a small number of different possible values, in ==regression the _target value is a ccontinuous value_, a real number==.   

The process of training a regressor can be completely different from training a classifier, but some of the algorithm for creating classification models can be modified to work with regression. 

## Linear regression
The starting point of regression is __linear regression__.
- Consider a dataset $X$ with $N$ rows and $D$ columns. 
- $x_i$ is a $D$ dimensional data element
- Consider then a response vector $\hat y$ with $N$ values $y_i$. 
- $w$ is a $D$-dimensional vector of coefficients that _needs to be learned_. 

- we ==model== the dependence of each response value $y_i$ from ==the corresponding independent variables== $x_i$ as
![[linear_regression.png]]
- such that the _error of modelling_ is _minimised_.

![[regression_visualization.png]]
We see that the easiest approximation of the data in the image is a line. 
The _score_ is obtained by computing the difference between each real value with its corresponding prediction, so it can formalized through an objective function that we wanto to minimize.

#### Objective function in regression
Our objective function that we want to minimize is:
![[regression_of.png]]
- Gradient of $O$ with respect to $w$: $2X^T(Xw^T - y)$
- By constraining the gradient to 0, we obtain the optimization condition:
$$
X^T Xw^T = X^T y
$$

If the symmetric matrix $X^TX$ is _invertible_ the solution can be derived as:
$$
w = (X^T X)^{-1}X^T y
$$
and the forecast is given by: $$y^f = X \cdot w^T$$
If $X^TX$ is not invertible, there could be issues. 

## Quality of the fitting
How we can measure the quality of the fitting? 
The most important quality measure is the __coefficient of determination__ $R^2$:
$$
R^2 = 1- \dfrac{SS_{res}}{SS_{tot}}
$$
Where:
![[some_regression_definitions.png]]

$R^2$ is an absolute number, and it compares the fit of the chosen model with that of a horizontal straight line. Despite the name, it isn’t the square of anything. 
If the model does not follow the trend of the data the numerator of the second term can reach or exceed the denominator. $R^2$ _can also be negative_.

We could also use the __Sum of squared residuals__ as a QM, but it is strictly related to the size of the data.  

Another surprising fact about $R^2$ is that we could see as the ratio between the actual error ($SS_{res}$) and the total sum of squares ($SS_{tot}$), subtracted from 1.

The _perfect value_ of $R^2$ would be _1_, since if $SS_{res} = 0$ then the prediction would not have any error whatsoever. 

###### Mean Squared Error and $R^2$
- Both refer to the error of the predictions, but...
- $R^2$ is a standardized index, 
- MSE measures the mean error, this it is influenced by the order of magnitude of the data.

### Multiple regression
Regression can be univariate, but also _multiple regression_ (meaning, with a vector of independent variables). 
- The response variable depends by more than one features 
- The regression technique is quite similar to that of simple regression 
![[multiple_regression_example.png]]
In this image, we have 2 independent variables x1 and x2, and a single response $y$.

In this situation, the estimation strategy is the same as the single regression. 

## Overfitting and Regularisation in Regression
- In presence of high number of features, _overfitting_ is possible, and performance on test data becomes much worse 
- _Regularisation_ reduces the influence of less interesting attributes and therefore reduces overfitting. 
	- \[not important\] There are regression techniques such as the Lasso regression that try to select the most interesting attribute and show us how things change if we would have used a small number of attributes, so that we can tune to our liking. 

## Polynomial regression
What if we have data that is not linear at all? (this can be possible even in a univariate situation).
![[polynomial_regression_examples.png]]

If a try a linear regression model, we could at most draw a line, like this:
![[underfitting_poly_regression.png]]
This obviously results in __underfitting__. 

So, the most obvious idea is to use, instead of a linear model, a _polynomial model_ (duh...).

Is the model still linear? Intuitively, we could say that a model is linear if the things that I want it to guess are linear, since we just want to guess the coefficients (which are linear values, i.e. with a exponent of 1). 
So, this is not so different from the general multivariate regression, we're just exploring a different perspective.  

In the case of polynomial regression, we can choose the order of our polynomial, and consequently the number of the coefficient that we want to use. 

Obviously, the higher the degree of the polynomial, the more accurate will be the approximation, but _it can also lead to overfitting_. 