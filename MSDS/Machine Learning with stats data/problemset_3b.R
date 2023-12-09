# PROBLEM SET 3B
# This assignment is done in a group of three
# Prajwal Kaushal, Sumukha Sharma, Aaryan Agarwal

###############################################################################
# Question 1
###############################################################################

# Loading data
data <- read.table("USAQ.txt", header = TRUE, sep = "\t")

# inspecting it by lookinng at the first few rows
head(data)

# checking the structure
str(data)

# summary
summary(data)

# subset of the data focusing on real interest rate(rf) and consumption growth(dc)
analysis_data <- data[, c("rf", "dc")]

# Summary for the subset
summary(analysis_data)

# Plotting the data to see relationship between dc and rf
plot(analysis_data$dc, analysis_data$rf, main = "Real Interest Rate vs Consumption Growth",
     xlab = "Consumption Growth (dc)", ylab = "Real Interest Rate (rf)")

###############################################################################
# Question 2
###############################################################################

# linear regression
model <- lm(rf ~ dc, data = analysis_data)

# summary of the regression model to interpret the coefficients
summary(model)

# Plotting the original data points
plot(analysis_data$dc, analysis_data$rf, main = "Linear Regression of RF on DC",
     xlab = "Consumption Growth (dc)", ylab = "Real Interest Rate (rf)")

# Adding the regression line to the plot
abline(model, col = "red")

# Intercept (0.0127557): This is the expected value of rf when dc is zero. 
# The intercept is significantly different from zero, as indicated by the p-value 
# (<2e-16, which is practically zero).

# dc Coefficient (-0.1723330): This is the estimated coefficient for consumption growth. 
# A negative coefficient shows that there is an inverse relationship between consumption 
# growth and the real interest rate.
# negative value implies risk lovinng behavior.

###############################################################################
# Question 3
###############################################################################

library(splines)

# cubic spline with one internal knot
median_dc <- median(analysis_data$dc)
model_spline_one_knot <- lm(rf ~ ns(dc, knots = median_dc), data = analysis_data)

# Plotting the cubic spline with the linear regression 
plot(analysis_data$dc, analysis_data$rf, main = "Cubic Spline vs Linear Regression",
     xlab = "Consumption Growth (dc)", ylab = "Real Interest Rate (rf)")
points(analysis_data$dc, predict(model_spline_one_knot), col = "red", type = "l")
abline(lm(rf ~ dc, data = analysis_data), col = "blue")

# cubic spline with five internal knots
quantiles_dc <- quantile(analysis_data$dc, probs = c(0.2, 0.4, 0.6, 0.8))
model_spline_five_knots <- lm(rf ~ ns(dc, knots = quantiles_dc), data = analysis_data)

# Plotting the data with the cubic spline (five knots)
plot(analysis_data$dc, analysis_data$rf, main = "Cubic Spline (Five Knots) vs Linear Regression",
     xlab = "Consumption Growth (dc)", ylab = "Real Interest Rate (rf)")
points(analysis_data$dc, predict(model_spline_five_knots), col = "red", type = "l")
abline(lm(rf ~ dc, data = analysis_data), col = "blue")

# Compare the variance of the one knot and five knot models
summary(model_spline_one_knot)
summary(model_spline_five_knots)

#summary(fit_splines)


# The coefficients of the cubic spline terms are mixed. 
# The residual standard error is 0.007224, which measures the standard deviation of the residuals.
# Multiple R-square is 0.03891, which means a slight improvement compared to the linear model.

# The coefficients are not statistically significant, as indicated by their p-values. 
# This suggests a more complex, possibly non-linear relationship.
# The residual standard error is 0.007249, slightly higher than the one-knot model, 
# => a marginal increase in the variance.
# Multiple R-squared is 0.04639, which is slightly higher than the one-knot model, 
# => a small improvement in the model's ability to explain the variability in rf.

# the slope is negative

###############################################################################
# Question 4
###############################################################################

# smoothing spline with cross-validation
smooth_spline_model <- smooth.spline(analysis_data$dc, analysis_data$rf, cv = TRUE)

# cubic spline with eight degrees of freedom
cubic_spline_model <- lm(rf ~ ns(dc, df = 8), data = analysis_data)

# Plotting the data with the smoothing spline and cubic spline fits
plot(analysis_data$dc, analysis_data$rf, main = "Smoothing Spline vs Cubic Spline (8 DF)",
     xlab = "Consumption Growth (dc)", ylab = "Real Interest Rate (rf)")
lines(smooth_spline_model, col = "red")
points(analysis_data$dc, predict(cubic_spline_model), col = "blue", type = "l")

# Interpretation based on the plot
# The smoothing spline (red line) shows a curve that deviates from a straight line, 
# => there are nonlinearities in the relationship between dc and (rf). 
# The cubic spline with eight degrees of freedom (blue lines) follows the data's minor
# fluctuations more closely than the smoothing spline. 
# This could indicate that the cubic spline is overfitting the data
# The overall slope of the smoothing spline seems to trend downwards as consumption growth increases.
# This would suggest risk-loving behavior.


