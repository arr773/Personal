# PROBLEM SET 2B
# This assignment is done in a group of three
# Prajwal Kaushal, Sumukha Sharma, Aaryan Agarwal

###############################################################################
#Question 1: Load, Prepare, and summarize the data
###############################################################################
library(hdm)
# This command loads data contained in a R-package.
data(cps2012)
?cps2012
# Construct a regressor matrix for use in the different models.
x <- model.matrix( ~ -1 + female + widowed + divorced + separated + nevermarried +
                     hsd08+hsd911+ hsg+cg+ad+mw+so+we+exp1+exp2+exp3, data=cps2012)
dim(x)

y <- cps2012$lnw


###############################################################################
#Question 2: Apply Ridge Regression With CV
###############################################################################

library(glmnet)

# Ridge regression
cv.ridge <- cv.glmnet(x, y, alpha=0, nfolds=10)
plot(cv.ridge)

# Optimal lambda
optimal_lambda_ridge <- cv.ridge$lambda.min
cat("Optimal Lambda for Ridge:", optimal_lambda_ridge, "\n")

# mse
ridge_optimal_mse <- cv.ridge$cvm[cv.ridge$lambda == optimal_lambda_ridge]
cat("Test MSE for Optimal Ridge:", ridge_optimal_mse, "\n")

# Number of variables
cat("Number of Variables in Ridge Regression:", length(coef(cv.ridge, s = optimal_lambda_ridge)), "\n")


# length(coef(cv.ridge, s = optimal_lambda_ridge)) 
# essentially counts the number of coefficients, including the intercept, 
# therefore it is 17 (16 predictors + 1 intercept).
# so there are 16 predictors.

# Ridge regression will include all predictors in the model; 
# it shrinks the coefficients but doesn't set any to zero. 
# So, the number of predictors remains the same as the original dataset, which is 16.

# Ridge regression adds a penalty to the size of coefficients, 
# which can prevent overfitting, especially when predictors are correlated.
# This regularization can lead to better out-of-sample predictions, 
# reducing the test MSE compared to OLS, which doesn't have this penalty.

# When lambda is zero, the penalty term disappears, and Ridge regression becomes 
# equivalent to OLS regression. That is, there's no regularization or shrinkage applied 
# to the coefficients.

# If the test MSE at the optimal lambda for Ridge regression is lower than the MSE
# for the Ridge regression at lambda = 0 (which is equivalent to OLS), then unrestricted OLS 
# is not optimal in terms of test MSE.


###############################################################################
#Question 3: Apply Lasso Regression With CV
###############################################################################

# Lasso regression
cv.lasso <- cv.glmnet(x, y, alpha=1, nfolds=10)
plot(cv.lasso)

# Optimal lambda
optimal_lambda_lasso <- cv.lasso$lambda.min
cat("Optimal Lambda for Lasso:", optimal_lambda_lasso, "\n")

# Coefficients at optimal lambda
coefficients_lasso <- coef(cv.lasso, s = optimal_lambda_lasso)

# Variables used
non_zero_coef <- coefficients_lasso[coefficients_lasso != 0]
cat("Number of Variables in Optimal Lasso Fit:", length(non_zero_coef) - 1, "\n")  # Subtracting intercept
print(non_zero_coef)

lasso_optimal_mse <- cv.lasso$cvm[cv.lasso$lambda == optimal_lambda_lasso]
cat("Test MSE for Optimal Lasso:", lasso_optimal_mse, "\n")


# Difference in MSE between Ridge and Lasso
difference_mse <- lasso_optimal_mse - ridge_optimal_mse
cat("Difference in MSE between Ridge and Lasso:", difference_mse, "\n")

# for this specific dataset and with the chosen predictor variables,
# the Lasso regression model generalizes slightly better to new, 
# unseen data than the Ridge regression model. However, it's worth noting that
# the difference is quite small. 
# In practical terms, the models might offer similar predictive performance.

# Since the dependent variable is the logarithm of the hourly wage, 
# the coefficient for female indicates that being female, on average, 
# is associated with a decrease in the logged hourly wage of about 0.28 units 
# compared to the baseline (which is male).
# Which means, being a female is associated with a wage that is exp(-0.27952186)
# or roughly 75.6% of the wage for males, holding all else constant. 
# This coefficient suggests a wage gap where females earn, on average, less than males 
# in this dataset.

###############################################################################
#Question 4: Using a more flexible model
###############################################################################


X <- model.matrix( ~ -1 +female+
                     female:(widowed+divorced+separated+nevermarried+
                               hsd08+hsd911+ hsg+cg+ad+mw+so+we+exp1+exp2+exp3) +
                     + (widowed + divorced + separated + nevermarried +
                          hsd08+hsd911+ hsg+cg+ad+mw+so+we+exp1+exp2+exp3)^2,
                   data=cps2012)
dim(X)
# Safety check: Exclude all constant variables.
X <- X[,which(apply(X, 2, var)!=0)]
dim(X)

index.gender <- grep("female", colnames(X))

# The provided code created a design matrix X that not only includes the main 
# effects of all predictors but also interaction effects of gender with all other 
# variables and squared terms for predictors.

# This new design matrix initially has 136 variables. 
# After removing constant variables, 116 remain.

# Ridge regression for new X
cv.ridge <- cv.glmnet(X, y, alpha=0, nfolds=10)
plot(cv.ridge)

# Optimal lambda
optimal_lambda_ridge <- cv.ridge$lambda.min
cat("Optimal Lambda for Ridge:", optimal_lambda_ridge, "\n")

# mse
ridge_flexible_mse <- cv.ridge$cvm[cv.ridge$lambda == optimal_lambda_ridge]
cat("Test MSE for Optimal Ridge:", ridge_flexible_mse, "\n")

# Number of variables
cat("Number of Variables in Ridge Regression:", length(coef(cv.ridge, s = optimal_lambda_ridge)), "\n")

# Lasso regression for new X

cv.lasso <- cv.glmnet(X, y, alpha=1, nfolds=10)
plot(cv.lasso)

# Optimal lambda
optimal_lambda_lasso <- cv.lasso$lambda.min
cat("Optimal Lambda for Lasso:", optimal_lambda_lasso, "\n")

# Coefficients at optimal lambda
coefficients_lasso <- coef(cv.lasso, s = optimal_lambda_lasso)

# Variables used
non_zero_coef <- coefficients_lasso[coefficients_lasso != 0]
cat("Number of Variables in Optimal Lasso Fit:", length(non_zero_coef) - 1, "\n")  # Subtracting intercept
print(non_zero_coef)
#print(coefficients_lasso)

lasso_flexible_mse <- cv.lasso$cvm[cv.lasso$lambda == optimal_lambda_lasso]
cat("Test MSE for Optimal Lasso:", lasso_flexible_mse, "\n")


# Difference in MSE between Ridge and Lasso
difference_mse <- lasso_flexible_mse - ridge_flexible_mse
cat("Difference in MSE between Ridge and Lasso:", difference_mse, "\n")


# Lasso regression has chosen 78 variables out of the total 116, 
# implying that lasso selected 38 variables to be not influential 
# in predicting wages in the context of other predictors. 

index.gender
#this shows that there are interaction term present with gender


###############################################################################
#Question 5: What is the most preferred prediction model of all?
###############################################################################


cat("Ridge MSE:", ridge_optimal_mse, "\n")
cat("Lasso MSE:", lasso_optimal_mse, "\n")
cat("Flexible Ridge MSE:", ridge_flexible_mse, "\n")
cat("Flexible Lasso MSE:", lasso_flexible_mse, "\n")

# here we can see that flexible lasso is the best model

print(coefficients_lasso)

# female:hsd08: The interaction between females and education level hsd08 is negative,
# suggesting that females with the hsd08 level of education might earn 0.0784 (or about 7.84%) 
# less on the log wage scale compared to the reference group, holding other predictors constant.

# female:hsd911: Females with education level hsd911 have a negative coefficient of 0.1517 
# (or about 15.17%), indicating they might earn less compared to the reference group.

# female:hsg: Females with high school graduation (hsg) education level have a negative 
# coefficient of 0.0226 (or about 2.26%), suggesting a wage reduction compared to the reference group.

# female:cg: The interaction coefficient for females with the cg level of education 
# is positive but quite small, suggesting a very slight increase in wages compared 
# to the reference group.

# female:ad: Females with the ad level of education have a negative coefficient of 0.0115 
# (or about 1.15%), indicating a slight wage reduction compared to the reference group.

# Conclusion:
# The flexible Lasso regression selected the interaction terms between gender (female) 
# and various education levels. The majority of these interactions suggest that females
# with these education levels have a reduction in wages when compared to the reference group 
# (which would be males with the same education level.
# The only exception is the cg education level, which indicates a slight wage 
# advantage for females.

