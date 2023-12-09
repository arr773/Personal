# PROBLEM SET 4a
# This assignment is done in a group of three
# Prajwal Kaushal, Sumukha Sharma, Aaryan Agarwal

#Part1
#Question 1
#1
calculate_na_proportion <- function(x) {
  mean(is.na(x))
}
# This function calculated the proportion of NA values in x

calculate_na_proportion(c(10,20,NA,40,50,NA,60,70))

standardize_vector <- function(x,na.rm = TRUE ) {
  x / sum(x, na.rm = na.rm)
}
#This function standardize the vector. It returns the normalize version where each element 
#is divided by the sum of all the elements in the vector so that it the sums to one. 
#The na.rm is used to handle the NA elements in the vector.
#If na.rm = FALSE, it returns NA and if na.rm = TRUE,ir drops NA 

#when na.rm = TRUE
standardize_vector(c(10,20,NA,40,50,NA,60,70))

#when na.rm = FALSE
standardize_vector(c(10,20,NA,40,50,NA,60,70),na.rm = FALSE )

calculate_coefficient_of_variation <- function(x) {
  sd(x, na.rm = TRUE) / mean(x, na.rm = TRUE)
}
#This funvtion calculates the coefficient of variation by dividing standard deviation with mean.
#Here both mean and standard deviation are calculated by dropping NA values since na.rm = TRUE

calculate_coefficient_of_variation(c(10,20,NA,40,50,NA,60,70))
###################################################################################################################
#2
both_na <- function(x,y) {
  if(length(x) == length(y)) {
    result <- is.na(x) & is.na(y)
  }
  return(result)
}

both_na(c(10,20,NA,40), c(NA,60,NA,NA))
###################################################################################################################
is_directory <- function(x) file.info(x)$isdir
is_readable <- function(x) file.access(x, 4) == 0

#The function is_directory checks whether the path in x is a directory.
#t uses the file.info function to retrieve file information and then extracts the "isdir" field from the result.

#The function is_readqble checks whether the file is readable which means 
#that the file in the given path exists  and the user has permission to open it.

#These two funvtions are useful while accessing the files, to check for the correct directory and
#user permisions to access the file.
###################################################################################################################
#4
f1 <- function(string, prefix) {
  substr(string, 1, nchar(prefix)) == prefix
}
#This function checks if each element in the vector statrs with the given prefix.
#the fucntion can be renamed as starts_with_prefix
starts_with_prefix <- function(string, prefix) {
  substr(string, 1, nchar(prefix)) == prefix
}
starts_with_prefix(c("abort","adapt","absorb","abroad","append"),"ab")

f2 <- function(x) {
  if (length(x) <= 1) return(NULL)
  x[-length(x)]
}
#This function drops the last element of the vector. If the vector has only one element or if the vector is
#empty, it returns NULL.the fucntion can be renamed as remove_last_element
remove_last_element <- function(x) {
  if (length(x) <= 1) return(NULL)
  x[-length(x)]
}
remove_last_element(c(1,2,3,4,5))
remove_last_element(c(1))
remove_last_element(c())


f3 <- function(x, y) {
  rep(y, length.out = length(x))
}
#This function repeats the element y element in the vector of lenght x.
#the fucntion can be renamed as repeat_element
repeat_element <- function(x, y) {
  rep(y, length.out = length(x))
}
repeat_element(c(1,2,3,4,5),3)
###################################################################################################################
#5
fizzbuzz <- function(x) {
  if((x %% 3 == 0) && (x %% 5 == 0 )){
    "fizzbuzz"
  }else if (x %% 3 == 0){
    "fizz"
  }else if (x %% 5 == 0){
    "buzz"
  }else {
    x
  }
}
fizzbuzz(15)
fizzbuzz(3)
fizzbuzz(10)
fizzbuzz(11)
###################################################################################################################
#6
temp <- 40
cut(temp, breaks = c(-Inf, 5, 10, 20, 30,Inf),
                          labels = c("freezing", "cold", "cool", "warm", "hot"),
                          right = TRUE)
#To use '<' instead of '<=' while working with cut, we can change right = FALSE
#chief advantage of using cut is that it can handle multiple values which means that it can work with vectors 
#which cannot happen while using if statement which takes single value. Also the handling the intervals for the 
#temperature values, we need to change the operators in case of if statement whereas with cut, we just need to
#change right = False.
#using cut also makes the code simple and readable.
###################################################################################################################
#7
x <-'e'
switch_out <- switch(x,
                     a = ,
                     b = "ab",
                     c = ,
                     d = "cd"
)
switch_out
#The switch function returns the first non missing values it encounters while matching the arguments.
#According to the above code, it will return 'ab' for 'a', 'ab' for 'b', 'cd' for 'c', 'cd' for 'd.' 
#If given 'e', it returns NULL since 'e' is not there in the given arguments.
###################################################################################################################
#Question 2
#1
#is.vector() checks if it is a specific type of vector with no attributes other than names. That is 
#is.vector() checks for a specific type of vector as defined by the specified mode and imposes 
#the constraint that the vector should have no attributes other than names.
#is.atomic() specifically checks if an object belongs to the atomic modes: "logical", "integer", 
#"numeric" (synonym "double"), "complex", "character", or "raw".If an object is of any of these 
#modes and has no attributes other than names, is.atomic() returns TRUE.
###################################################################################################################
#2
x <- c(5,10,15,20,25,30)

#returns last element
last_ele <- function(x) {
  x[length(x)]
}
last_ele(x)

#elements at even numbered positions
even_positions <- function(x){
  x[seq(from = 2, length(x),by = 2)]
}
even_positions(x)

#Every element except the last value.
except_last_value <- function(x){
  x[1:length(x)-1]
}
except_last_value(x)

#Only even numbers
even_numbers <- function(x){
  x[x %% 2 == 0]
}
even_numbers(x)
##################################################################################################################
#3
#x[-which(x > 0)] and x[x <= 0] gives similar results. However, the difference comes while handling null values
#x[-which(x > 0)] will ignore `NA`'s and leave them as it is and 
# x[x <= 0]  returns any value that cannot be comparable as NA
x <- c(1,2,3,4,5,NaN,NA)
x[-which(x > 0)]
x[x <= 0]
##################################################################################################################
#4
#when we subset with a positive integer that’s bigger than the length of the vector, It returns NA
x <- c(2,4,6,8)
x[5]

#when we subset with a name that doesn’t exist, still it returns NA
x <- c(a = 1, b = 2, c = 3)
x["d"]
##################################################################################################################
#5
res = list('a', 'b', list('c', 'd'), list('e', 'f'))
res1 = list(list(list(list(list(list('a'))))))
##################################################################################################################
#Question 3
#1
#he mean of every column in mtcars
library(tidyverse)
mtcars_means <- vector("double", ncol(mtcars))
names(mtcars_means) = names(mtcars)
for (i in names(mtcars)) {
  mtcars_means[i] <- mean(mtcars[, i])
}
mtcars_means

#type of each column in nycflights13::flights
library(nycflights13)
flights_types <-  vector("list", ncol(flights))
names(flights_types) <- names(flights)
for (i in names(flights)){
  flights_types[[i]] <-  class(flights[[i]])
}
flights_types

#number of unique values in each column of iris
data(iris)
iris_unique_counts <- vector("double", ncol(iris))
names(iris_unique_counts) <- names(iris)
for (i in names(iris)) {
  iris_unique_counts[i] <- length(unique(iris[[i]]))
}
iris_unique_counts

#10 random normals for each of mu = -10, 0, 10, and 100
mu_values <- c(-10, 0, 10, 100)
random_normals <- vector("list", length(mu_values))
for (i in seq_along(random_normals)) {
  random_normals[[i]] <- rnorm(10, mean = mu_values[i])
}
random_normals
###################################################################################################################
#2
out <- ""
for (x in letters) {
  out <- str_c(out, x)
}
out
#str_c works with vectors. Hence we can use str_c() with the collapse to return single argument
str_c(letters, collapse = "")

x <- sample(100)
sd <- 0
for (i in seq_along(x)) {
  sd <- sd + (x[i] - mean(x)) ^ 2
}
sd <- sqrt(sd / (length(x) - 1))
sd
#we have inbuilt function for standard deviation and we can use that
sd(x)

x <- runif(100)
out <- vector("numeric", length(x))
out[1] <- x[1]
for (i in 2:length(x)) {
  out[i] <- out[i - 1] + x[i]
}
out
#The code is calculating cumulative sum which can be done using cumsum()
all.equal(cumsum(x), out)
###################################################################################################################
#3
output <- vector("integer", 0)
for (i in seq_along(x)) {
  output <- c(output, lengths(x[[i]]))
}
output

#To discuss the performance effect, we are definig two functions and use the microbenchmark package 
#to compare the time 
add_to_vector <- function(n) {
  output <- vector("integer", 0)
  for (i in seq_len(n)) {
    output <- c(output, i)
  }
  output
}
add_to_vector_2 <- function(n) {
  output <- vector("integer", n)
  for (i in seq_len(n)) {
    output[[i]] <- i
  }
  output
}
library(microbenchmark)
timings <- microbenchmark(add_to_vector(10000), add_to_vector_2(10000), times = 3)
timings
#From the results we can say that the pre-allocated vector performs much faster. However, the longer the vector
#and the bigger the objects, the more that pre-allocation will outperform appending

##################################################################################################################
#Part2
#RMarkdown is widely appreciated for its ability to integrate code, text, and visualizations in a single document.
#I find it valuable for creating reproducible and dynamic reports, especially in statistical modeling projects.
#I also appreciate the ability to render the document in various formats (e.g., HTML, Word, PDF) 
#allowing for flexibility and customization.
#Once missing feature which I came across is that the feacture facilitating collaborative writing and editing
#which makes beneficial while working in teams.












