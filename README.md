Download Link: https://assignmentchef.com/product/solved-elen4720-problem-2
<br>
Problem 1

In this problem you will derive a naive Bayes classifier. For a labeled set of data (<em>y</em><sub>1</sub><em>,x</em><sub>1</sub>)<em>,…,</em>(<em>y<sub>n</sub>,x<sub>n</sub></em>), where for this problem <em>y </em>∈ {0<em>,</em>1} and <em>x </em>is a <em>D</em>-dimensional vector of counts, the Bayes classifier observes a new <em>x</em><sub>0 </sub>and predicts <em>y</em><sub>0 </sub>as

<em>y</em><sub>0 </sub>= argmax<em>.</em>

The distribution <em>p</em>(<em>y</em><sub>0 </sub>= <em>y</em>|<em>π</em>) = Bernoulli(<em>y</em>|<em>π</em>). What is “naive” about this classifier is the assumption that all <em>D </em>dimensions of <em>x </em>are independent. Assume that each dimension of <em>x </em>is Poisson distributed with a Gamma prior. The full generative process is

<em>iid</em>

Data: <em>y<sub>i </sub></em>∼ Bern(<em>π</em>)<em>, x<sub>i,d</sub></em>|<em>y<sub>i </sub></em>∼ Pois(<em>λ<sub>y</sub></em><em><sub>i</sub></em><em>,d</em>)<em>, d </em>= 1<em>,…,D </em>Prior:Gamma(2<em>,</em>1) Derive the solution for <em>π </em>and each <em>λ<sub>y,d </sub></em>by maximizing

<em>π,  </em>= arg          max                                                                                                                                                 <em>.</em>

<em>π,λ</em>

Please separate your derivations as follows: (a) Derive <em>π </em>using the objective above.

b

(b) Deriveusing the objective above, leaving <em>y </em>and <em>d </em>arbitrary in your notation.

Problem 2

In this problem you will implement the naive Bayes classifier derived in Problem 1, as well as the kNN algorithm and logistic regression algorithm. The data consists of examples of spam and non-spam emails, of which there are 4600 labeled examples. The feature vector <em>x </em>is a 54-dimensional vector extracted from the email and <em>y </em>= 1 indicates a spam email.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>

In every experiment below, <em>randomly </em>partition the data into 10 groups and run the algorithm 10 different times so that each group is held out as a test set one time. The final result you show should be the cumulative result across these 10 groups.

<ul>

 <li>Implement the naive Bayes classifier described above. In a 2 × 2 table, write the number of times that you predicted a class <em>y </em>data point (ground truth) as a class <em>y</em><sup>0 </sup>data point (model prediction) in the (<em>y,y</em><sup>0</sup>)-th cell of the table, where <em>y </em>and <em>y</em><sup>0 </sup>can be either 0 or 1. There should be four values written in the table in your PDF. Next to your table, write the prediction accuracy—the sum of the diagonal divided by 4600. (The sum of all entries in the table should be 4600.)</li>

 <li>In one figure, show a stem plot (stem() in Matlab) of the 54 Poisson parameters for each class averaged across the 10 runs. (This average is only used for plotting purposes on this homework. In practice you would relearn these parameters using the entire data set to find their final values.) Use the README file to make an observation about dimensions 16 and 52.</li>

 <li>Implement the <em>k</em>-NN classifier for <em>k </em>= 1<em>,…,</em>20. Use the <em>`</em><sub>1 </sub>distance for this problem. Plot the prediction accuracy as a function of <em>k</em>.</li>

</ul>

Problem 3

In this problem you will implement the Gaussian process model for regression. You will use the same data used for homework 1 to do this, which is again provided in the data zip file for this homework. Recall that the Gaussian process treats a set of <em>N </em>observations (<em>x</em><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<em>x<sub>N</sub>,y<sub>N</sub></em>), with <em>x<sub>i </sub></em>∈ R<em><sup>d </sup></em>and <em>y<sub>i </sub></em>∈ R, as being generated from a multivariate Gaussian distribution as follows,

use: <em>.</em>

Here, <em>y </em>is an <em>N</em>-dimensional vector of outputs and <em>K </em>is an <em>N </em>× <em>N </em>kernel matrix. For this problem use the Gaussian kernel indicated above. In the lecture slides, we discuss making predictions for a new <em>y</em><sup>0 </sup>given <em>x</em><sup>0</sup>, which was Gaussian with mean <em>µ</em>(<em>x</em><sup>0</sup>) and variance Σ(<em>x</em><sup>0</sup>). The equations are shown in the slides.

There are two parameters that need to be set for this model as given above, <em>σ</em><sup>2 </sup>and <em>b</em>.

<ol>

 <li>Write code to implement the Gaussian process and to make predictions on test data.</li>

 <li>For <em>b </em>∈ {5<em>,</em>7<em>,</em>9<em>,</em>11<em>,</em>13<em>,</em>15} and <em>σ</em><sup>2 </sup>∈ {<em>.</em>1<em>,.</em>2<em>,.</em>3<em>,.</em>4<em>,.</em>5<em>,.</em>6<em>,.</em>7<em>,.</em>8<em>,.</em>9<em>,</em>1}—so 60 total pairs (<em>b,σ</em><sup>2</sup>)— calculate the RMSE on the 42 test points as you did in the first homework. Use the mean of the Gaussian process at the test point as your prediction. Show your results in a table.</li>

 <li>Which value was the best and how does this compare with the first homework? What might be adrawback of the approach in this homework (as given) compared with homework 1?</li>

</ol>

To better understand what the Gaussian process is doing through visualization, re-run the algo-rithm by using <em>only </em>the 4th dimension of <em>x<sub>i </sub></em>(car weight). Set <em>b </em>= 5 and <em>σ</em><sup>2 </sup>= 2. Show a scatter plot of the data (<em>x</em>[4] versus <em>y </em>for each point). Also, plot as a solid line the predictive mean of the Gaussian process at each point <em>in the training set</em>. You can think of this problem as asking you to create a test set by duplicating <em>x<sub>i</sub></em>[4] for each <em>i </em>in the training set and then to predict that test set

<a href="#_ftnref1" name="_ftn1">[1]</a> I’ve preprocessed the data. The original data is at https://archive.ics.uci.edu/ml/datasets/Spambase. More information about the meanings of the 54 dimensions of the data is provided in two accompanying files.