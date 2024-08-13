#import libraries for statistical analysis later
library(MASS)
library(tweedie)
library(e1071)
library(rpart.plot)
library(randomForest)
library(gplm)
library(caret)
library(gbm)

#import data into R from MASS library
data(Boston)
#print out head of dataframe 
head(Boston)

#plot density function for target variable
hist(Boston$medv,prob=TRUE,
     main="Loss Density",
     xlab="Loss",col="darkgreen", 
     border="white",
     breaks = 40,
     xlim =c(0,55))
#plot continuous pdf
lines(density(Boston$medv), col = "red")

#Only categorical variable in dataset is CHAS - no need for encoding

#plot out Tweedie Distribution with different alpha
# Define a sequence of x values to evaluate the PDF
x <- seq(0, 10, length.out = 1000)
# Set up the parameters for the Tweedie distribution
p_values <- c(1.2, 5, 10)
# Define colors for the different lines
colors <- c("blue", "red", "green")

# Initialize the plot with the first set of parameters
plot(x, dtweedie(x, mu = 1, phi=1, power = p_values[1]), 
     type = "l", lwd = 2, col = colors[1],
     main = "Tweedie PDFs for Different Parameters",
     xlab = "x", ylab = "Density",
     ylim = c(0, 2))  # Adjust ylim based on your data
# Overlay additional lines for the other parameter combinations
for (i in 1:length(p_values)){
  lines(x,dtweedie(x,mu=1,phi=1,power = p_values[i]),col=colors[i],lwd=2,lty=i)
}
# Add a legend
legend("topright", legend = paste("p =", p_values),
       col = colors, lwd = 2, lty = 1:length(p_values))
#notice that spike around 0 is higher as power is higher for Tweedie
#notice that our data does not have a spike at 0, Tweedie may not be a good choice

#plot out lognormal distribution with different alpha
# Define a sequence of x values to evaluate the PDF
x <- seq(0, 10, length.out = 1000)
# Initialize the plot with the first set of parameters
plot(x, dlnorm(x), 
     type = "l", lwd = 2, col = colors[1],
     main = "Lognormal Distribution",
     xlab = "x", ylab = "Density",
     ylim = c(0, 0.8))  # Adjust ylim based on your data
#Seems that shape of lognormal looks similar to the loss distribution- we can try to fit the GLM with a log link function

#split Boston dataset into train and test set
#set seed for reproducibility
set.seed(123)
# Determine the sample size for the training set (e.g., 80% of the data)
train_indices <- sample(1:nrow(Boston), size = 0.8 * nrow(Boston))
# Split the data into training and test sets
train_data <- Boston[train_indices, ]  
test_data <- Boston[-train_indices, ]
# Check the dimensions of the training and test sets
dim(train_data)
dim(test_data)

#notice that data available for us to work with is quite scarce (400 observations in train set, therefore we need to use weaker models)

#fit glm (log link)
logNormGLM=glm(medv~., data=train_data,family = gaussian(link="log"))
summary(logNormGLM)
#generate prediction for test set observations
logNormGLM_pred=exp(predict(logNormGLM,newdata=test_data[,-which(names(test_data)=="medv")],interval="prediction"))
plot(x=test_data$medv,y=logNormGLM_pred,main="y_predicted vs y_actual for test set",xlab="Actual",ylab="Predicted")
abline(0, 1, col = "red")  # Add a 45-degree reference line
#calculate MSE
logNormGLMerr=logNormGLM_pred-test_data$medv
print(paste("MSE for lognormal GLM:",mean(logNormGLMerr^2)))

#fit SVM
#Normalize train data and extract parameters trained on train set and use to scale test set
std_scaler = scale(train_data[,-which(names(train_data) == "medv")])
train_data_normalized = as.data.frame(std_scaler)
# Add the 'medv' column back to the scaled training data
train_data_normalized$medv <- train_data$medv

# Extract location and scale parameters from train set 
norm_loc=attr(std_scaler, "scaled:center")
norm_sd=attr(std_scaler, "scaled:scale")

#use extracted parameters to scale test set
test_data_normalized = as.data.frame(scale(test_data[,-which(names(test_data) == "medv")], center = norm_loc, scale = norm_sd))
# Add the 'medv' column back to the normalized test data
test_data_normalized$medv=test_data$medv

#perform kfold CV for SVM
#set k for CV
k=5
tune_grid <- expand.grid(
  sigma = seq(0.01, 0.2, length = 5),  # Example sigma values
  C = seq(6, 30, length = 5)          # Example C values
)
cv_control=trainControl(method="cv",number=k)
cvSVM=train(medv~.,data=train_data_normalized,method="svmRadial",trControl=cv_control,tuneLength=10,tuneGrid=tune_grid)
print(cvSVM)
print(cvSVM$bestTune)
plot(cvSVM)
#In general, for low levels of sigma (scaling for distance- implying we give less weight to observations further away)
#high levels of regularization perform better with low sigma (give less weight to faraway points- improves prediction and higher regularization on weights- reduces variance in prediction)
#we choose sigma=0.06 and C=30 (svm performance significantly better with sigma around 0.06 with both C=12/C=30 performing similarly)
#fit svm
SVM=svm(medv~., data=train_data_normalized,gamma=0.06,cost=30,scale=FALSE)
summary(SVM)
#predict on test set
SVM_pred=predict(SVM,newdata=test_data_normalized[,-which(names(test_data_normalized)=="medv")])
#plot predicted vs actual
plot(x=test_data$medv,y=SVM_pred,main="y_predicted vs y_actual for test set",xlab="Actual",ylab="Predicted")
abline(0, 1, col = "red")  # Add a 45-degree reference line
#calculate MSE
SVMerr=SVM_pred-test_data$medv
print(paste("MSE for SVM:",mean(SVMerr^2)))
#the performance is better than the results from the LR with log link function
#However,complicated model does not always yield better results

#perform t test to compare the two models
t.test(logNormerr,SVMerr)
#as we can see from the t tests, the two samples are not significantly different
#perform Wilcoxon rank sum test to compare the two models
wilcox.test(logNormerr,SVMerr)
#results similar to the t test, samples not significantly different

#try to fit a decision tree
dt=rpart(medv~.,data=train_data,method="anova")
summary(dt)
prp(dt)
dt_pred=predict(dt,newdata=test_data[,-which(names(test_data)=="medv")],type="vector")
#plot predicted vs actual
plot(x=test_data$medv,y=dt_pred,main="y_predicted vs y_actual for test set",xlab="Actual",ylab="Predicted")
abline(0, 1, col = "red")  # Add a 45-degree reference line
#calculate MSE
dterr=dt_pred-test_data$medv
print(paste("MSE for decision tree:",mean(dterr^2)))
#the performance is worse than SVM
#complicated model does not always yield better results

#try to fit a random forest
rf=randomForest(medv~.,data=train_data)
print(rf)
rf_pred=predict(rf,newdata=test_data[,-which(names(test_data)=="medv")])
#plot predicted vs actual
plot(x=test_data$medv,y=rf_pred,main="y_predicted vs y_actual for test set",xlab="Actual",ylab="Predicted")
abline(0, 1, col = "red")  # Add a 45-degree reference line
#calculate MSE
rferr=rf_pred-test_data$medv
print(paste("MSE for random forest:",mean(rferr^2)))
#the performance is worse than SVM
#complicated model does not always yield better results

#try to fit a gradient boosted tree
GBM=gbm(medv~.,data=train_data)
print(GBM)
GBM_pred=predict(GBM,newdata=test_data[,-which(names(test_data)=="medv")])
#plot predicted vs actual
plot(x=test_data$medv,y=GBM_pred,main="y_predicted vs y_actual for test set",xlab="Actual",ylab="Predicted")
abline(0, 1, col = "red")  # Add a 45-degree reference line
#calculate MSE
GBMerr=GBM_pred-test_data$medv
print(paste("MSE for Gradient Boosted Tree:",mean(GBMerr^2)))
#the performance is better than decision tree but worse than random forests

#create table to compare results
results=data.frame(Model=c("Linear Regression(Lognormal Link)","Linear Regression(Gamma Link)""SVM","Decision Tree","Random Forest (500 trees default)","Gradient Boosted Tree"),MSE=c(mean(logNormerr^2),mean(SVMerr^2),mean(dterr^2),mean(rferr^2),mean(GBMerr^2)))
print(results)
