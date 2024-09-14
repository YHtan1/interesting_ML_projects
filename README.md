# interesting_ML_projects
| Projects| Links |
| ------ | ------ |
| STAT4609 Projects |[Projects for STAT4609 Big Data Analytics](STAT4609_Projects) |
| Self-studying ML | [Review of ML Models](ML_ModelReview) |
| Self - Written ML Algorithms | [Implementation of Interesting ML Algorithms](self_written_algo) |
|Boston Housing Dataset|[Study of different ML models on the Boston Housing Dataset](BostonHousing)|
|Titanic dataset|[Predicting Survivors in the Titanic Disaster](titanic_dataset)|

# **STAT4609 Projects**
[Projects for STAT4609 Big Data Analytics](STAT4609_Projects)

**Applied OOP concepts in designing and implementing ML algorithms from scratch in Numpy for STAT 4609 Big Data Analytics.**

<details>
  <summary>Detailed Explanation of STAT4609 Projects</summary>

# 1. HW1
[STAT4609_Projects_HW1](STAT4609_Projects/hw1_v2.ipynb)

- Implementation of **linear regression** from scratch using **Numpy**
- Implementation of **ridge regression** and **cross validation** for hyperparameter tuning for regularization hyperparameter

# 2. HW2
[STAT4609_Projects_HW2](STAT4609_Projects/hw2_v5.ipynb)

- Implementation of **logistic regression** and **Naive Bayes classifier** from scratch in **Numpy**

# 3. HW3
[STAT4609_Projects_HW3](STAT4609_Projects/hw3_v5.ipynb)

- Implemented **decision tree** from scratch in Numpy and fitted on iris dataset
- Extended algorithm to run **random forest** on iris dataset

# 4. HW4 
[STAT4609_Projects_HW4_GibbsSampler](STAT4609_Projects/hw4_v7.ipynb)

[STAT4609_Projects_HW4_ExpectationMaximization](STAT4609_Projects/hw4_v5_final.ipynb)

- Implemented **K-Nearest Neighbours (KNN)** model on synthetic dataset from scratch
- 1. Implemented **Gaussian Mixture Model (GMM)** on dataset and fitted it using **Expectation Maximization (EM)** algorithm
- 2. Implemented **Gaussian Mixture Model** on dataset fitted it using **Gibbs Sampler**
- **This project was _extremely difficult_ as EM was prone to blowing up due to bad initialization.**
- The Gibbs Sampler also was unable to converge to a desirable solution and I realized that you needed to perform **"thinning"** on the sampled parameters as the sampled parameters are correlated since it is a special case of **MCMC**.
- I had to simulate much more samples and average the **i\*100 th samples**, where i=1,2,3.. to reduce the correlation between samples and improve my estimate of the parameters. 
- This made me realize that **Monte Carlo simulation** for option pricing is extremely  computationally expensive as its standard deviation reduces by a factor of 10 as you increase the number of iterations by a factor of 100! _(std estimator=std of samples/sqrt(n))_

# HW5 (Final Project)
[STAT4609_Projects_HW5_Belkor](STAT4609_Projects/final_project_v4.ipynb)

- Implemented **Belkor** on the **Movie Lens** dataset to generate recommendations for moviegoers (Belkor was the winning submission of a Neflix Movie Reccomendation Competition)  
- It is difficult to use traditional machine learning algorithms as the data for movie goers are **extremely sparse**. 
- Forget about using NNs as the counts of ratings have a **long right tail** - most movies have little ratings and there is insufficient data to fit a NN model.
- The dataset is also extremely huge, making it difficult to use computationally expensive models as recommendations must be generated ASAP, making neighbourhood-based collaborative filtering a good choice.
</details>

# **ML_ModelReview**
[Review of ML Models](ML_ModelReview)

**This section contains Machine Learning/ Deep Learning models that I have implemented**

<details>
  <summary>Detailed Description of ML Projects</summary>

# 1. Review of Linear Regression
[Review of Linear Regression and its extensions](ML_ModelReview/Linear_model_review.ipynb)

- Review of Linear Regression models and some of its extensions, for example ridge/LASSO/LAR and Forward/Backward Selection 

# 2. Trees vs NN 
[Working File - Comparison of Tree-based Models vs Neural Networks](ML_ModelReview/MLP_v2.ipynb)

**Study of different ML models on the MNIST dataset of handwritten digits**
Although this dataset is not interesting (studies have shown that we can virtually get a near 100% classification rate on MNIST - further info in [Wiki link](https://en.wikipedia.org/wiki/MNIST_database)), this project demonstrates that sometimes the elementary models (**Random Forest** or **Decision Trees**), perform similarly to more flexible models (**NN**) and require **less computational power**.

**CNN** performs better than **MLP** with half the layers and much less computational time, showing that the **CNN** stucture is a superior model to the vanilla **MLP**.

Although **Tree-based** methods show comparable performance, **NN** are still superior to them as we can feed data to **NN** in batches, avoiding memory constraints. As GPUs are used for training models nowadays, **NN** also benefit from parallel computing vs **Tree-based** methods, which are trained sequentially.  

</details>

# **Self-written algo**
[Implementation of Interesting ML Algorithms](self_written_algo)
**Implementation of ML algorithms unavailable online that I wrote after reading about them in textbooks!**

<details>
  <summary>Detailed Description of Self-Written Algorithms</summary>
  
# 1. Successive Orthogonalization
[Implementation of Interesting ML Algorithms](self_written_algo)

</details>

# **BostonHousing**
[Study of different ML models on the Boston Housing Dataset](BostonHousing)
**A report on different ML models on the Boston Housing Dataset.**

<details>
  <summary>Detailed Description of BostonHousing</summary>
  
# 1. Boston Housing
[R code](https://github.com/YHtan1/interesting_ML_projects/blob/main/BostonHousing/Boston_final.R)
[Report](https://github.com/YHtan1/interesting_ML_projects/blob/main/BostonHousing/Boston.pdf)
- This report reviews different commonly used machine learning models like GLM/LR, SVM, tree-based methods like GBM/ Random Forests, etc. 
- This report provides a detailed demonstration on how to fit them correctly on data to extract insight on the dataset reviewed. 
- We will help build intuition on the unique characteristics of each model and how it affects the learning of data. 
</details>


# **Titanic Dataset**
[Predicting Survivors in the Titanic Disaster](titanic_dataset)
# **Attempt to predict survivorship status for people aboard the RMS Titanic**
<details>
  <summary>Detailed Description of Titanic Dataset</summary>

# 1. titanic_graphs
[Graphs](titanic_dataset/titanic_graphs.ipynb)
- In this file, I perform an in-depth analysis of the data, starting with plotting out the correlation heatmap of the data.
- I also plot the survival rates for each category for categorical features to investigate the effect of each individual feature on survivability.
- I also split the continuous numeric features into bins to improve predictability (reduces the number of splits that can be made based on values in that column-avoids overfitting)

# 2. titanic_final
[Finalized Models](titanic_dataset/titanic_final.ipynb)
- Stores the finalized models for submission
- Decided to use the XGBoost Random Forest as my final submission model - earning me a 78.23% accuracy rate (top 20% percentile)
- Best submissions score around 80% - quite satisfied with my results
- Also tried vanilla XGBoost and stacking (XGBoost+SVC->Logistic Regression) - however score is similar to XGBoost RF

# 3. Lesson Learnt from this Kaggle Competition
- Learnt data cleansing skills using pandas/ scikit-learn 
- Decision Trees may get higher scores than RF on small datasets (suspect that bootstrap samples for RF may not capture observations from rare cases) 
but variance between CV scores can be larger than decision trees - unable to generalize well to unseen test set
- Feature engineering is extremely important-binning the continous variables + grouping certain rare categories together often helps wiht performance
- Maybe Bayesian methods to weight estimators for a category vs estimator based on whole dataset will help reduce overfitting? (Worth further investigation)


</details>
