# PROBLEM SET 3A
# This assignment is done in a group of three
# Prajwal Kaushal, Sumukha Sharma, Aaryan Agarwal


#Part 1: R Questions

###############################################################################
#Question 1: Data Import and tidying
###############################################################################

###############################################################################
# TIBBLES ANND DATA FRAMES

#1. We can tell if an objest is a tibble in 2 ways:
# -> By printing:  When you print a tibble, it shows the first 10 rows and 
# all columns that fit on the screen, along with the type of each column. 
# This is different from regular data frames, which may print many more rows 
# and potentially not fit well on the screen.
# -> Class check: Using the class() function in R. 
# For a tibble, it will return "tbl_df", "tbl", and "data.frame", 
# indicating it's a tibble and a data frame. 
# For a regular data frame like mtcars, it will just return "data.frame".

#2. 
df <- data.frame(abc = 1, xyz = "a")
df$x
df[, "xyz"]
df[, c("abc", "xyz")]

# df$x: Accessing a column.
# data.frame: Returns the column x. If x doesn't exist, it returns NULL.
# tibble: Same behavior as a data.frame.
# df[, "xyz"]: Extracting a single column.
# data.frame: Returns a vector.
# tibble: Returns a tibble with one column. 
# This is a key difference and can be more consistent for programming 
# since the output type is predictable.
# df[, c("abc", "xyz")]: Extracting multiple columns.
# data.frame: Returns a data.frame with the specified columns.
# tibble: Similar behavior, but with tibble specific formatting and printing.

###############################################################################
# DATA IMPORT

#1. 
# we cann use read_delim() from the readr package in R. 
# This function allows you to specify any delimiter, including "|".
# Example: read_delim(file, delim = "|")

#2.
# read_fwf() is used for reading fixed width files. 
# The important arguments are:
# file: The path to the file or a connection.
# col_positions: This is crucial as it defines the start and end points of each 
# column. we can use fwf_positions(), fwf_widths(), or fwf_empty() to specify this.
# col_types: To specify the type of each column. 
# It helps in controlling how columns are read and preventing unnecessary type conversions.
# Other common arguments like col_names, na, skip, etc., can also be important 
# depending on the specific needs of the data import.

#3. 
# To read text like "x,y\n1,'a,b'" using read_delim(), we need to specify the 
# delimiter and the quote character. 
# Since the data is using a comma as a delimiter and a single quote (') for quoting, 
# the command would be:
# read_delim("x,y\n1,'a,b'", delim = ",", quote = "'")
# Here, delim = "," specifies that the fields are separated by commas, 
# and quote = "'" tells the function to recognize single quotes as the quoting character, 

###############################################################################
# PARSING VECTORS

#1. If we set both decimal_mark and grouping_mark to the same character in R, 
# it will result in an error because the parser won't be able to differentiate between 
# the decimal point and the thousands separator. 
# This would make parsing numbers ambiguous.
# When we set decimal_mark to ",", the default value of grouping_mark typically changes to "." 
# in many international ares where a comma is used as a decimal separator. 
# Conversely, if we set grouping_mark to ".", the default decimal_mark often becomes ","

#2. 
# Europe: UTF-8 is widely used due to its ability to handle a vast range of characters 
# and its compatibility with various languages and systems. 
# Other encodings like ISO-8859-1 is also used
# Asia: Some common encodings include Big5, GB 2312, GBK, and HZ. 
# These encodings are particularly used for Chinese characters. 
# Other Asian languages also have specific encodings, 
# but the widespread adoption of Unicode and UTF-8 has significantly reduced the 
# dependency on these region-specific encodings.

###############################################################################
# SPREADING AND GATHERING

#1.
library(tibble)
library(tidyr)
library(dplyr)
stocks <- tibble(
  year = c(2015, 2015, 2016, 2016),
  half = c( 1, 2, 1, 2),
  return = c(1.88, 0.59, 0.92, 0.17)
)
stocks %>%
  spread(year, return) %>%
  gather("year", "return", `2015`:`2016`)

# In the given example, the spread() function transforms the stocks tibble by 
# spreading the year column into multiple columns, each named after a specific year 
# and containing values from the return column. 
# However, when you reverse the process with gather(), 
# the year becomes a character type instead of numeric because column 
# names are always characters. 
# This change in data type is one reason why gather() and spread() are not 
# perfectly symmetrical.

#2. 
# The convert argument attempts to automatically convert the result columns to 
# the appropriate data type. By default, when spreading or gathering, the columns 
# are character type. 
# When convert is set to TRUE, it tries to infer and convert the data types
# based on the data content.

#3.
# The code fails because gather() expects column names or indices as the second 
# and third arguments, but '1999' and '2000' are interpreted as numbers, 
# not as column names.

#4.
# The spreading fails because there are duplicate combinations of name and key. 
# For eg. "Phillip Woods" has two entries for "age". 
# To fix this, we can add a new column that creates a unique identifier for each row
# before spreading.

#5.
preg <- tribble(
  ~pregnant, ~male, ~female,
  "yes", NA, 10,
  "no", 20, 12
)

# Transforming the data into a tidy format
tidy_preg <- preg %>%
  gather(key = "gender", value = "count", -pregnant)

tidy_preg

# The variables are "pregnant", "gender", and "count". 
# After gathering, you would have a tidy format where each row is an observation 
# with a "pregnant" status, "gender" (male or female), and the corresponding count.

###############################################################################
# SEPARATING AND UNITING
#1.

# extra and fill are arguments in the separate() function that control how 
# additional pieces of the split string are handled.

tibble(x = c("a,b,c", "d,e,f,g", "h,i,j")) %>%
  separate(x, c("one", "two", "three"))
tibble(x = c("a,b,c", "d,e", "f,g,i")) %>%
  separate(x, c("one", "two", "three"))


# First dataset experiment
tibble(x = c("a,b,c", "d,e,f,g", "h,i,j")) %>%
  separate(x, c("one", "two", "three"), extra = "merge", fill = "right")

# For the first dataset, extra = "merge" will merge any extra pieces into 
# the last column, and fill = "right" will fill missing values from the right with NA.

# Second dataset experiment
tibble(x = c("a,b,c", "d,e", "f,g,i")) %>%
  separate(x, c("one", "two", "three"), extra = "drop", fill = "right")

# For the second dataset, extra = "drop" will drop any extra pieces, 
# and fill = "right" again fills missing values from the right with NA.

#2. 
# The remove argument determines whether the original columns that are being 
# united or separated should be removed from the resulting data frame.
# When set to TRUE (default), the original columns are removed.
# When set to FALSE, the original columns are retained alongside the new columns.

###############################################################################
# MISSING VALUES

#1. 

# The fill argument in spread() is used to replace NA values that appear in 
# the spread data. When spreading a key-value pair across a wider format, 
# any missing combinations will result in NA. 
# The fill argument allows to specify a value that should replace these NAs.
# In complete(), the fill argument also replaces NA values,
# but it does so in a different context. 
# complete() is used to expand a dataset to include all combinations 
# of specified keys, filling in NA where data does not exist. 
# The fill argument in complete() lets us specify values to replace these NAs 
# across the newly created rows.

#2. 

# The direction argument in the fill() function specifies the direction in 
# which to fill missing values (NA).
# The options are:
# down: Fills values downwards (from top to bottom).
# up: Fills values upwards (from bottom to top).
# downup: First fills downwards, then upwards. 
# This is useful when you want to fill NAs with the 
# nearest non-NA value either above or below.
# updown: First fills upwards, then downwards.

###############################################################################
#Question 2: Relational Data and Data Types
###############################################################################

###############################################################################
# RELATIONAL DATA

# 1.
# Variables Needed: We need the geographical coordinates (latitude and longitude) 
# of both the origin and destination airports.
# Tables Needed: You would combine data from the flights table (which includes 
# origin and destination airport codes) with the airports table 
# (which provides the coordinates of each airport).

# 2. 
# The relationship is likely based on the location. 
# The weather data would correspond to the airports based on the airport's 
# geographical location. 

# 3.
# If the weather table contained records for all airports in the US, it would 
# define an additional relationship with the destination airports in the flights table. 
# This means there would be two relationships for weather: one with the origin airport
# and another with the destination airport in the flights table.

# 4.
# we can create a data frame with dates and an indicator of whether the day is special 
# (e.g., a holiday).
# The primary key would be the date.
# This table could be connected to the flights table through the date, 
# allowing analysis of flight patterns on these special days 

###############################################################################
#  KEYS

#1.

library(nycflights13)

flights_with_key <- flights %>%
  mutate(surrogate_key = row_number())

# In this code the mutate() function is used to add a new column called 
# surrogate_key to the flights table. 
# The row_number() function generates a sequence of numbers from 1 to 
# the number of rows in the table, ensuring each row has a unique identifier.


###############################################################################
# MUTATING JOINS

#1.

library(ggplot2)
# install.packages("maps")
library(maps)
library(nycflights13)

airports %>%
  semi_join(flights, c("faa" = "dest")) %>%
  ggplot(aes(lon, lat)) +
  borders("state") +
  geom_point() +
  coord_quickmap()

# Calculate average delay by destination
avg_delay_by_dest <- flights %>%
  group_by(dest) %>%
  summarize(avg_delay = mean(arr_delay, na.rm = TRUE))

# Join with airports data
delay_airports <- airports %>%
  semi_join(flights, c("faa" = "dest")) %>%
  inner_join(avg_delay_by_dest, by = c("faa" = "dest"))

# Plot the map
ggplot(delay_airports, aes(x = lon, y = lat, size = avg_delay, color = avg_delay)) +
  borders("state") +
  geom_point() +
  coord_quickmap()

# 2.

flights_with_loc <- flights %>%
  left_join(airports, c("origin" = "faa")) %>%
  rename(origin_lat = lat, origin_lon = lon) %>%
  left_join(airports, c("dest" = "faa")) %>%
  rename(dest_lat = lat, dest_lon = lon)

# 3.

# Joining flights and planes tables
flights_planes_joined <- flights %>%
  left_join(planes, by = "tailnum")

# Calculating the age of the planes using the 'year.y' column
flights_planes_joined <- flights_planes_joined %>%
  mutate(plane_age = 2013 - year.y) %>%  # Using 'year.y' for plane manufacture year
  filter(!is.na(plane_age))  # Removing rows where plane age could not be calculated

# Analyzing the relationship between plane age and delays
correlation_result <- cor(flights_planes_joined$plane_age, flights_planes_joined$arr_delay, use = "complete.obs")

correlation_result
# A correlation coefficient of approximately -0.01767 suggests a very weak negative 
# relationship between the age of the plane and its delays

#4.

# Filter flights on June 13, 2013
june_13_flights <- flights %>%
  filter(month == 6, day == 13, year == 2013) %>%
  left_join(airports, c("dest" = "faa"))

# Plotting
ggplot(june_13_flights, aes(x = lon, y = lat, color = arr_delay)) +
  borders("state") +
  geom_point() +
  coord_quickmap()


###############################################################################
# FILTERING JOINS

#1.

# Counting flights per plane
plane_flight_count <- flights %>%
  group_by(tailnum) %>%
  summarize(flight_count = n()) %>%
  filter(flight_count >= 100)


plane_flight_count

# Filtering flights
flights_with_frequent_planes <- flights %>%
  semi_join(plane_flight_count, by = "tailnum")

flights_with_frequent_planes

#2.

# Finding top 48 hours with worst delays
top_delays <- flights %>%
  group_by(year, month, day, hour) %>%
  summarize(total_delay = sum(dep_delay, na.rm = TRUE)) %>%
  arrange(desc(total_delay)) %>%
  slice_head(n = 48)

top_delays

# Joining with weather data
weather_delays <- top_delays %>%
  left_join(weather, by = c("year", "month", "day", "hour"))

weather_delays

#3.

# Checking if each plane is associated with a single airline
plane_airline_relation <- flights %>%
  group_by(tailnum) %>%
  summarize(airlines = n_distinct(carrier)) %>%
  filter(!is.na(tailnum))

# Checking for planes associated with more than one airline
multi_airline_planes <- plane_airline_relation %>%
  filter(airlines > 1)

multi_airline_planes

# There are 17 planes in the nycflights13 dataset that are associated with more than one airline,
# as each of these planes has an airlines count of 2. 
# This finding rejects the hypothesis that each plane is exclusively flown by a single airline. 
# In this dataset, at least, some planes are operated by multiple airlines.

###############################################################################
# STRINGS

#1. 
# paste() concatenates strings with a separator between them. By default, this separator is a space.
# paste0() is a variation of paste() that uses an empty string as the separator, 
# effectively concatenating strings without any space.
# In the stringr package, the equivalent function is str_c().
# Regarding NA handling: paste() turns NA into "NA" (a string), while paste0() does the same. 
# In contrast, str_c() will return NA if any of the inputs is NA, unless na.rm = TRUE is specified.

#2.
# The sep argument specifies the string to use between each element when concatenating.
# The collapse argument is used when combining multiple strings into a single string. 
# It specifies the separator to use between the combined strings.

#3.
# str_wrap() wraps a string into formatted lines of a specified width. This is useful for 
# creating text with a specific width for display purposes, such as in console output or 
# when formatting text for reports.

#4.
# str_trim() trims whitespace from the start and end of a string.
# The opposite function could be considered to be adding spaces or padding to a string, which can be done using 
# functions like str_pad() in stringr. However, this is not a direct opposite as str_pad() requires specification
# of the desired string length and padding character.

###################################################################################################################

#Part 2 - Project

##################################################################################################################
#An interesting data set we came across was the General Social Survey (GSS)
#dataset. It is a high-quality survey which gathers data on American society
#and opinions, and it conducted since 1972. Our research delves into understanding the impact of socio-economic 
#factors on educational attainment in urban and rural settings.
#The primary question revolves around identifying the key determinants that influence 
#educational outcomes, particularly focusing on the influence of household income, parental education, 
#and geographical location. The data set can be accessed in
#R using the library infer. The dataset present in R is a sample of the
#original dataset of 500 entries from the original with a span of years
#1973-2018.It includes demographic markers and some economic variables.
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

summary(data)

#Data collected after the year 2000
newdata1<-filter(data,year>2000)
plot(newdata1['sex'])

plot(newdata1['partyid'])

plot(newdata1['class'])


#Data collected after the year 2000 and the survey weight is greater than 1
newdata2<-filter(data,year>2000 & weight>1)
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


