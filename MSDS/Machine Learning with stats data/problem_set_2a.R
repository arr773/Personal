# PROBLEM SET 2A
# This assignment is done in a group of three
# Prajwal Kaushal, Sumukha Sharma, Aaryan Agarwal

###############################################################################
#PART 1: R QUESTIONS
###############################################################################
#Question 1: Data Transformation
###############################################################################
# Filtering
library(nycflights13)
sum(is.na(flights$dep_time))
colSums(is.na(flights))
# we can see that 8255 flights have missing dep_time
# There are other variables that have missing values like - dep_delay, arr_time, 
# arr_delay, tailnum, air_time
# Rows with missing dep_time probably represent cancelled flights or flights that didn’t 
# get recorded properly.

# 2.

# NA ^ 0 gives 1 because anything raised to the power of 0 is 1.
#NA | TRUE gives TRUE because OR plus anything with TRUE is always TRUE.
#FALSE & NA gives FALSE because AND plus anything with FALSE is always FALSE.
#General rule:

# For arithmetic operations: If the result of the operation is determinable 
# irrespective of the actual value of the missing data (NA), R will return that determinate value. 
# For example:
# Any number to the power of zero is 1, so NA ^ 0 returns 1, irrespective of what NA represents.
# Multiplication involving 0 will always result in 0, irrespective of the other operand. 
# therefore, NA * 0 gives 0.
# For logical operations:
# If an operation involves OR and one of the values is TRUE, the outcome is definitely 
# TRUE irrespective of the other operand's value. So, NA | TRUE gives TRUE.
# If an operation involves AND and one of the values is FALSE, the outcome is definitely FALSE 
# irrespective of the other operand's value. Thus, FALSE & NA gives FALSE.

# ARRANGE DATA:
library(dplyr)
# 1.
dplyr::arrange(nycflights13::flights, desc(is.na(dep_time)))

# 2. 
# For the longest flight
longest_flight <- dplyr::arrange(nycflights13::flights, desc(distance)) %>% dplyr::slice(1)
longest_flight

# For the shortest flight
shortest_flight <- dplyr::arrange(nycflights13::flights, distance) %>% dplyr::slice(1)
shortest_flight

# SELECT COLUMNS:

# 1.

dplyr::select(nycflights13::flights, year, year, year)
# as we can see herethe column will only be selected once 



# Create new variables
# 1.


flights$difference <- flights$arr_time - flights$dep_time
selected_flights <- flights[, c("air_time", "dep_time", "arr_time", "difference")]
head(selected_flights)

# air_time is the time spent in the air, 
# while arr_time - dep_time is a rough measure of the total duration of the flight. 
# However, this isn't a perfect comparison because arr_time and dep_time are in the HHMM format, 
# while ait_time is in minutes, 
# so direct subtraction will lead to incorrect calculations. 
# To fix this, we need to convert these times into a format that correctly represents the duration.
# something like this:
# Computing the minutes since midnight for departure and arrival
flights$dep_minutes <- (flights$dep_time %/% 100) * 60 + (flights$dep_time %% 100)
flights$arr_minutes <- (flights$arr_time %/% 100) * 60 + (flights$arr_time %% 100)

# 2. 

selected_columns <- flights[, c("dep_time", "sched_dep_time", "dep_delay")]
head(selected_columns)

# dep_delay is the difference between dep_time and sched_dep_time. 
# So, dep_time = sched_dep_time + dep_delay.

# 3.

1:3 + 1:10

# This gives a warning. 
# The reason is that R is trying to repeat the shorter vector 
# to match the length of the longer one. 
# It repeats 1:3 until it reaches the length of 1:10


# Grouped summaries

# 1.

# I am assuming that the dep_time in na for cancelled flights
# Grouping by year, month, and day
grouped_flights <- group_by(flights, year, month, day)
daily_summary <- summarise(grouped_flights, 
                           cancelled_flights = sum(is.na(dep_time)),
                           avg_delay = mean(dep_delay, na.rm = TRUE))
print(daily_summary, n=100)

library(ggplot2)

ggplot(daily_summary, aes(x = avg_delay, y = cancelled_flights)) +
  geom_point(aes(color = as.Date(paste(year, month, day, sep = "-"))), alpha = 0.6) +  # color by date for additional info
  geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "dashed") +  # adds a linear regression line
  labs(title = "Relationship between Average Delay and Cancelled Flights",
       x = "Average Delay (in minutes)",
       y = "Number of Cancelled Flights",
       color = "Date") +
  theme_minimal()


# we can see that on many days where there's a high number of cancelled flights, 
# there's also a high average delay. Therefore, there seems to be a relationship 
# between the proportion of cancelled flights and the average delay.
# with the plot we can say that it seems to be kind of linear.

# 2. 
# carriers with worst delays
carrier_delays <- flights %>%
  group_by(carrier) %>%
  summarise(
    avg_delay = mean(arr_delay, na.rm = TRUE)
  ) %>%
  arrange(desc(avg_delay))

head(carrier_delays)


# Challenge
carrier_airport_counts <- flights %>%
  group_by(carrier, dest) %>%
  summarise(
    flights_count = n(),
    avg_delay = mean(arr_delay, na.rm = TRUE)
  ) %>%
  arrange(desc(avg_delay))

head(carrier_airport_counts)
# from this grouped summary we ccan seee which combination of carriers and destinations 
# have the most flights and the average delay. 
# we have to think about these two questions:
# Are there carriers that consistently have bad delays irrespective of the destinations?
# Are there destinations that consistently have delays irrespective of the carrier?


###############################################################################
#Question 2: Exploratory Data Analysis
###############################################################################

# Typical and atypical values

library(ggplot2)

# 1.

ggplot(diamonds, aes(x=x)) + geom_histogram(binwidth=0.2) + ggtitle("Distribution of x")
ggplot(diamonds, aes(x=y)) + geom_histogram(binwidth=0.2) + ggtitle("Distribution of y")
ggplot(diamonds, aes(x=z)) + geom_histogram(binwidth=0.2) + ggtitle("Distribution of z")

# 2. 

ggplot(diamonds, aes(x=price)) + geom_histogram(binwidth=10) + 
  ggtitle("Distribution of Price with binwidth = 10")

ggplot(diamonds, aes(x=price)) + geom_histogram(binwidth=100) + 
  ggtitle("Distribution of Price with binwidth = 100")

ggplot(diamonds, aes(x=price)) + geom_histogram(binwidth=500) + 
  ggtitle("Distribution of Price with binwidth = 500")
# By adjusting the binwidth, we can observe the granularity of the distribution.
# we can see a huge spike between 500-800
# there is almost no bar around 1400 price

# 3. 
sum(diamonds$carat == 0.99)
sum(diamonds$carat == 1)

# A 1-carat diamond might be seen as more prestigious than a 0.99-carat diamond, 
# influencing purchasing and selling behavior.

# Missing values

# 1.

# Missing values are ignored in histograms and won't be represented in the bins. 
# In bar charts, NA is considered as just another category and they could get their own bar.

# 2.
# This removes NA values from the vector before calculating the mean and sum


# Covariance

# 1.

# using hisstogram
ggplot(diamonds, aes(x=carat)) + geom_histogram(aes(fill=..count..), binwidth=0.05) + facet_wrap(~cut(price, breaks=5)) + ggtitle("Distribution of Carat Partitioned by Price")
# This visualizes how the distribution of carat changes across different price segments

# using boxplot
ggplot(diamonds, aes(x = cut_number(price, 10), y = carat)) +
  geom_boxplot() +
  labs(title = "Distribution of Carat by Price Bins",
       x = "Price Bins",
       y = "Carat") +
  theme_minimal()


# 2.

large_diamonds <- diamonds[diamonds$carat > 2, ]
small_diamonds <- diamonds[diamonds$carat <= 0.5, ]

ggplot(large_diamonds, aes(x=price)) + geom_histogram(binwidth=100) + ggtitle("Price Distribution of Large Diamonds")
ggplot(small_diamonds, aes(x=price)) + geom_histogram(binwidth=100) + ggtitle("Price Distribution of Small Diamonds")

# The distribution of large diamonds is variable
# The of large diamond is higher.


###############################################################################
#PART 2: Project
###############################################################################

#An interesting data set we came across was the General Social Survey (GSS)
#dataset. It is a high-quality survey which gathers data on American society
#and opinions, and it conducted since 1972. The data set can be accessed in
#R using the library infer. The dataset present in R is a sample of the
#original dataset of 500 entries from the original with a span of years
#1973-2018. It includes demographic markers and some economic variables.
#It contains of 11 variables namely year (year the respondent was surveyed),
#age (age of the respondent at the time of the survey), sex (gender of the
#respondent which is self-identified by them), college 
#(whether the respondent has a valid college degree or no),
#partyid (respondents political party affiliation), 
#hompop (number of people in the respondents house), 
#hours (number of hours the respondent works while he was being surveyed),
#income (total family income of the respondent), class 
#(subjective socioeconomic class identification), finrela 
#(opinion of family income) and weight (survey weight). The data set consists 
#of just 500 rows of data.
#We can use this dataset to generate the average number of people living
#in each household in a certain year. We can chart out the slope of the '
#increase or the decrease in the number of people in each household. 
#We can determine how much an average worker works each week and 
#the average salary they get for each hour. We can group the previous
#result based on the class of the individual. We can determine which political
#party is likely to succeed in that area during a specific year. The literacy 
#rate of the area can be determined on whether a person has achieved a degree 
#or not. Many such inferences can be made through this dataset by various 
#statistical methods. We can group the dataset based upon the years by 
#splitting the dataset and can determine many inferences according to the year. 
#Same can be done by splitting the dataset by class or political party
#preferences.
#Its good data because we can infer many different conditions as given above 
#and it gives us a lot of potential. The original dataset is available on the
#gss website and should be easily accessible. A shorter format is available 
#in the infer library in R if for some reason we are not able to process the
#data.
library(tidyverse)
library(infer)
library(ggplot2)
data<-gss
newdata1<-filter(data,year>2000)
#Data collected after the year 2000
plot(newdata1['sex'])
plot(newdata1['partyid'])
plot(newdata1['class'])
newdata2<-filter(data,year>2000 & weight>1)
#Data collected after the year 2000 and the survey weight is greater than 1
plot(newdata2['sex'])
plot(newdata2['partyid'])
plot(newdata2['class'])
#We can see that the females have a higher survey weight than the men
#Data collected from men and women respectively
newdata3<-filter(data,year>2000 & sex=='female')
plot(newdata3['partyid'])
newdata4<-filter(data,year>2000 & sex=='male')
plot(newdata4['partyid'])
#We can see that the females tend to vote for the democratic party
#less than the males
#Data collected from people above and below the age of 35 respectively
newdata5<-filter(data,year>2000 & age>35)
plot(newdata5['partyid'])
newdata6<-filter(data,year>2000 & age<=35)
plot(newdata6['partyid'])
#We can see that the people below the age of 35 have less confidence
#in the democratic party than the people above the age of 35
