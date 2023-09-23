# This assignment is done in a group of three
# Prajwal Kaushal, Sumukha Sharma, Aaryan Agarwal

# Part a
#Question 1
data=read.csv("D:/Programs/Personal/MSDS/Machine Learning with stats data/Advertising.csv")
data[1:5,]
#removing the first row as it is just the index
data=data[,2:5]
data[1:5,]
#summary of data
summary(data)
#plotting the data
plot(data)

#Question 2
#Linear Regression between Sales and TV
model1=lm(Sales~TV, data = data)
summary(model1)
#R sq 0.609 is a good fir
#Linear Regression between Sales and Radio
model2=lm(Sales~Radio, data = data)
summary(model2)
#R sq 0.328 is a avg fit
#Linear Regression between Sales and Newspaper
model3=lm(Sales~Newspaper, data = data)
summary(model3)
#R sq 0.047 is a bad fit



#Question 3
#Multiple Linear Regression
model4=lm(Sales~TV+Newspaper+Radio,data=data)
summary(model4)
#R sq 0.895 is a good fit
library(rgl)
library(car)
scatter3d(Sales~TV+Radio,data=data)

#Question 4
# Multiple Regression with Interaction Term between TV and Radio
model5=lm(Sales ~ TV * Radio, data=data)
summary(model5)
#R sq 0.967 is a very good fit
#Better fit than previous models

#Other Interaction terms
model6=lm(Sales ~ TV * Newspaper, data=data)
summary(model6)
#R sq 0.643 good fit

model7=lm(Sales ~ Radio * Newspaper, data=data)
summary(model7)
#R sq 0.33 avg fit

model8=lm(Sales ~ TV * Radio * Newspaper, data=data)
summary(model8)
#R sq 0.967 is a very good fit

#Question 5
#Linear Regression between Sales and (TV+Radio+TV:Radio)
model9=lm(Sales ~ TV + Radio + TV:Radio , data=data)
summary(model9)

beta1=coef(model9)["TV"]
beta1
beta2=coef(model9)["Radio"]
beta2
beta3=coef(model9)["TV:Radio"]
beta3

# optimal values of TV and Radio
newTV=(beta1-beta2+300*beta3)/(2*beta3)
newTV
newRadio=300-newTV
newRadio
newdata=data.frame(TV=newTV,Radio=newRadio)
newdata
#optimal sales 
newSales=predict(model9,newdata = newdata)
newSales
#confidence interval for the prediction
ci=predict(model9, newdata=newdata, interval="confidence")
ci

# Part B

# What is the goal of Machine Learning?
# Machine learning specialists are often primarily concerned with developing
# high-performance computer systems that can provide useful predictions in the
# presence of challenging computational constraints.
# It offers a set of tools that can usefully summarize various sorts
# of nonlinear relationships in the data.
# The goal of machine learning is typically to achieve good out-of-sample predictions.
# In other words, the aim is to build models that perform well on new, unseen data,
# rather than just on the data they were trained on.

####################################################################################
# What does Varian mean by "good out-of-sample predictions"?
# "Good out-of-sample predictions" refers to the ability of a model to generalize
# well to new data that it hasn't seen before (data that was not used in training the model).
# A model that makes accurate predictions on new data has good out-of-sample performance

####################################################################################
# What is overfitting?
# Overfitting occurs when a model is too complex and fits the training data too closely,
# capturing even its noise. Such a model will perform poorly on new, unseen data
# because it has become too tailored to the training set.

####################################################################################
# What is model complexity?
# Model complexity refers to the number of parameters in a model or the intricacy of its structure.
# A more complex model might fit the training data very well but may not generalize well to new 
# data,leading to overfitting. Varian suggests that if we have a numeric measure of model complexity,
# we can view it as a parameter that can be adjusted or "tuned" to achieve the best out-of-sample
# predictions.

####################################################################################
# What is the training data?
# Training data is used to estimate or train a model. In the process of building a model,
# data is typically split into training, validation, and testing sets.
# The model is trained on the training data, the best model structure or hyperparameters are
# chosen using the validation data, and the model's performance is evaluated on the testing data.


