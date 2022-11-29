---------
#### Intro

##### Table of contents
1. [[#Introduction|Introduction]]
	1. [[#Knowledge vs Data|Knowledge vs Data]]
	2. [[#Introduction#Some vocabulary|Some vocabulary]]
	3. [[#Some vocabulary#The Discovery Process in Data Mining (Map)|The Discovery Process in Data Mining (Map)]]
	4. [[#Some vocabulary#The Machine Learning Map|The Machine Learning Map]]
2. [[#Some vocabulary#Tasks to "solve" in the process|Tasks to "solve" in the process]]
3. [[#Supervised vs unsupervised methods|Supervised vs unsupervised methods]]
	1. [[#How to obtain supervised information?|How to obtain supervised information?]]
4. [[#Supervised vs unsupervised methods#Reinforcement Learning (a few words)|Reinforcement Learning (a few words)]]
5. [[#Supervised vs unsupervised methods#Decision Process|Decision Process]]
6. [[#Supervised vs unsupervised methods#Model of data|Model of data]]

###### _Lesson resources_. 
Slides [here](https://virtuale.unibo.it/pluginfile.php/1336016/mod_resource/content/7/machineLearning-01-intro.pdf).

--------------

# Introduction 

### Knowledge vs Data
 - __Data__ are facts about reality or a certain entity
 - __Knowledge__ is information extrapolated from said data

## Some vocabulary
- __Data Mining__ – The entire discovery process sets up a pipeline starting from the data and ending with the general patterns and their translation into useful actions. 
	- From this whole process, we acquire knowledge. 
- __Machine Learning__ – the application of methods and algorithms <u>to extract the patterns from the data</u>.

### The Discovery Process in Data Mining (Map)
![[DiscoveryProcessMap.png]]
### The Machine Learning Map
![[MachineLearningMap.png]]

### Tasks to "solve" in the process
(this list can be seen as an index of what we will see in the next weeks).
Here are some of the tasks to do during the data mining process. 

- _Classification_ and _class probability estimation_: 
	- among the customers of a phone company, which are likely to respond to a given offer? can I sort them on the basis of the probability of responding? 
	- 
- __Regression (value estimation)__ 
	- given a set of numeric attribute values for an individual, estimate the value of another numeric attribute. 
		- i.e. How much will a given customer, whose characteristics are known, use a given service? 
	- it is related to classification, ==but the methods are completely different==.
	- It is a supervised activity

- __Similarity matching__: identify similar individuals based on data known about them
	- necessitates similarity measures
		- e.g. which are the companies similar to our best customers? they could be target of our next customer acquisition campaign 


- __Clustering__: groups individual in a population on the basis of their similarities.
	- i.e. DNA sequences could be clustered in [functional groups](https://en.wikipedia.org/wiki/Functional_group) 

- __Co–occurrence grouping__: attempts to find associations between entities according to the transactions in which the appear together 
	- (also known as frequent item set mining, association rule discovery, market basket analysis) 
	- i.e. what items are commonly purchased together?

- __Profiling__ (also known as behavior description) 
	- i.e. What is the typical cell phone usage of this customer segment? 
	- the behavior can be described in a complex way over an entire population 
	- usually the population is divided in groups of similar behavior
	- useful also to detect _anomalies_ 

- __Link analysis and prediction__: in a world where there exist items and connections (i.e. a graph), try to <abbr title="dedurre">infer</abbr> missing connections from the existing ones
	- since you and Karen share ten friends, maybe you would like to be Karen’s friend?

- __Data reduction__: attempts to take a large set of data and replace it with a reduced one, ==preserving most of the important information==.
	- a smaller set can be easier to manipulate, or even show more general insights
	- involves _loss of information_. 
	- looks for the best trade–off between information loss and improved insight 

- __Causal modeling__: understand what events or actions actually influence others 
	- i.e. consider that we use predictive modeling to target advertisements to consumers, and we observe that indeed the targeted consumers purchase at a higher rate after having been targeted. Was this because the advertisements influenced the consumers to purchase? Or did the predictive models simply do a good job of identifying those consumers who would have purchased anyway?


# Supervised vs unsupervised methods

Let's consider these questions: 
- Does our population _naturally_ fall into different groups? 
	- When asking this question, there is no specific purpose or target for grouping: it should emerge by observing the characteristics of the individuals. 
	- This is an example of ==__unsupervised mining__==.
- Can we find groups of customers who have particularly high likelihoods of canceling their service soon after their contracts expire? 
	- <u>We have just defined a specific target </u>: cancelling or not.
	- This is an example of ==__supervised mining__==. 
	- This problem is called _churn analysis_.

In summary: 
- supervised → specific target
- _unsupervised_ → natural process, without a specific target

The techniques for supervised situations are <u>substantially different</u> from those of the unsupervised ones. 
Being supervised or unsupervised is a characteristic of the problem and/or the data, it is not a design choice.
Supervised information is usually added to the attributes of the individuals.

##### How to obtain supervised information?
There are two main ways to obtain supervised information: 
 - information provided by experts e.g. the soybean disease labels of the example of page 
 - history 
	 - e.g. we have an history of the customers who cancelled their service subscription. 
	 - the supervised information is not available run–time, when we must decide what to do.
	 - later on the history will tell us the value of the unknown attribute which influences our actions.
	 - we want to learn how to guess the unknown attribute from the known ones.

## Reinforcement Learning (a few words) 
- target: __a sequence of actions__ which obtains the best result
	- it is not a label, or a group, or a value, or a class
- learn: a policy 
- how: try a policy – get a reward – change the policy focus: the overall policy, rather than the single actions

[orphan notes...]
## Decision Process
The __decision process__ 

## Model of data
A __model of data__ is 

In this case, the model is the analisys that the soft
