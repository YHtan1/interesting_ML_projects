# STAT4609_published
This Github page stores the ML algorithms I wrote for STAT 4609 Big Data Analytics
I am familiar with the concepts of Object Oriented Programming as I had the chance to implement such principles in this course

hw1
1. code to perform linear regression from scratch using numpy
2. code for ridge regression from scratch using numpy and also implemented cross validation to find the best hyperparameter for regularization

hw2
1. code for logistic regression in numpy
2. code to implement Naive Bayes classifier in numpy

hw3
1. wrote decision tree code from scratch to fit on iris dataset
2. implemented random forest to fit model on iris dataset
   
hw4
1. implemented KNN algorithm on synthetic dataset from scratch
2. fit Gaussian Mixture Model on dataset using EM algorithm
3. fit Gaussian Mixture Model on dataset using Gibbs Sampler
This project was extremely difficult as EM was prone to blowing up due to bad initialization. I had to spend a lot of time to troubleshoot my project and read through many reference books and papers in order to solve this problem
Gibbs Sampler also faced similar problems as I only realized that you needed to perform "thinning" on after fitting the model as the sampled parameters have correlation between them since it is MCMC. I had to simulate much more samples and average the i*100 th samples, where i=1,2,3....... to get a good estimate of the parameters.
Linking to my other projects on option pricing, Monte Carlo simulation needs a lot of iterations to get a good estimation and this is extremely computationally expensive- its std reduces by a factor of 10 as you increase the number of iterations by a factor of 100!!!

hw5
1. Implemented Belkor on the Movie Lens dataset to generate recommendations for moviegoers
It is difficult to use traditional machine learning algorithms as the data for movie goers are extremely sparse. Forget about using neural networks as the counts of ratings have a right tail- most movies have little ratings and there is insufficient data to fit a neural network model.
The dataset is also extremely huge, making it difficult to use computationally expensive models as recommendations must be generated ASAP, making neighbourhood-based collaborative filtering a good choice. 
