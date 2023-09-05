-- Aaryan Agarwal Fall 2023

select * from teachers;

select last_name,first_name,salary from teachers;

select distinct school from teachers;

select distinct school,salary from teachers;

select first_name,last_name,salary from teachers order by salary desc;

select last_name,school,hire_date from teachers order by school asc,hire_date desc;

select last_name,school,hire_date from teachers where school='Myers Middle School';

select first_name,last_name,school from teachers where first_name='Janet';

select school from teachers where school!='F.D. Roosevelt HS';

select first_name,last_name,hire_date from teachers where hire_date<'2000-01-01';

select first_name,last_name,salary from teachers where salary>=43500;

select first_name,last_name,salary from teachers where salary between 40000 and 65000;

select first_name from teachers where first_name like 'sam%';

select first_name from teachers where first_name like binary 'sam%';

select * from teachers where school='Myers Middle School' and salary <40000;

select * from teachers where last_name='Cole' or last_name='Bush';

select * from teachers  where school='F.D. Roosevelt HS' and (salary < 3800 or salary>40000);

select first_name,last_name,hire_date,salary from teachers where school like '%Roos%' order by hire_date desc;