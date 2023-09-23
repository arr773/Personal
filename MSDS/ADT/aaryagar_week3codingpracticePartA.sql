# Aaryan Agarwal Fall 2023

# Remember you need to select the query you are executing, then query > execute (or icon - lightning strike)
SELECT actor_id,first_name
FROM actor;

# Select 5 TOP names
SELECT last_name
FROM actor
LIMIT 5;

#Count how many rows
SELECT count(*)
FROM actor;

# What is 5% from 200? 
SELECT (5/200)*100;

# we can select 3 top records - about 5%
SELECT last_name
FROM actor
LIMIT 3;

# What are unique first names?
SELECT DISTINCT first_name
FROM actor;

# Add ALIAS
SELECT first_name AS actor_name
FROM actor;

## PART 1 ORDER and FILTER
# Ascending order
SELECT title, release_year
FROM film
ORDER BY release_year
LIMIT 5; 

# Descending order
SELECT title, release_year
FROM film
ORDER BY release_year DESC, title ASC
LIMIT 5;

# Conditions
SELECT customer_id, amount
FROM payment
WHERE amount < 5
LIMIT 5;

SELECT customer_id, amount
FROM payment
WHERE amount <> 0.99 
LIMIT 5;

SELECT customer_id, amount
FROM payment
WHERE amount BETWEEN 1 and 3#
LIMIT 5;

SELECT customer_id, amount
FROM payment
WHERE amount NOT BETWEEN 1 and 3#
LIMIT 5;

## NULL VALUES
select *   
FROM film
LIMIT 5;

SELECT title
FROM film
WHERE original_language_id IS NULL
LIMIT 5;

SELECT title
FROM film
WHERE description IS NOT NULL
LIMIT 5; 

## PART 1 WHERE Conditions
SELECT title, rental_rate
FROM film
WHERE rating > 4
limit 5;

SELECT title, rental_rate
FROM film
WHERE rating = 'PG' AND rental_rate > 4
limit 5;

# OR
SELECT title, rating
FROM film
WHERE rental_rate = 4.99 OR rental_rate = 2.99
limit 5;

# IN
SELECT title, rating
FROM film
WHERE rental_rate IN (4.99, 2.99)
limit 5;

SELECT title
FROM film
WHERE title LIKE 'T%'
LIMIT 5;

### PART 2 Aggregate
SELECT SUM(amount) 
FROM payment;

SELECT SUM(amount) as total_amount
FROM payment;

# COUNT
SELECT COUNT(film_id) 
FROM inventory;

SELECT COUNT(DISTINCT film_id) 
FROM inventory;

# MIN MAX
SELECT MIN(amount)
FROM payment;

SELECT MAX(amount) 
FROM payment;

SELECT AVG(amount) 
FROM payment;

### PART 2 Group By and HAVING
SELECT rating, COUNT(title) as titles_count
FROM film
GROUP BY rating;

SELECT rating, COUNT(title) as titles_count
FROM film
WHERE rental_rate > 4
GROUP BY rating;

SELECT rating, COUNT(title) as titles_count
FROM film
WHERE rental_rate > 4
GROUP BY rating
HAVING COUNT(title) > 70;

## PART 3 JOIN
SELECT first_name, last_name, rental_id
FROM customer
INNER JOIN payment ON customer.customer_id = payment.customer_id
LIMIT 5;

SELECT first_name, last_name, payment.rental_id, amount
FROM customer
INNER JOIN payment ON customer.customer_id = payment.customer_id
INNER JOIN rental ON payment.rental_id = rental.rental_id
LIMIT 5;

## PART 3 LEFT JOIN
SELECT first_name, last_name, film_id
FROM actor
LEFT JOIN film_actor ON film_actor.actor_id = actor.actor_id
LIMIT 5;

SELECT first_name, last_name, film_id
FROM actor
RIGHT JOIN film_actor ON film_actor.actor_id = actor.actor_id
LIMIT 5;

### PART 3 UNION
SELECT first_name, last_name, active, store_id
FROM customer
WHERE store_id IN (1,2)
UNION ALL
SELECT first_name, last_name, active, store_id
FROM customer
WHERE store_id IN (4,5)
LIMIT 5;

