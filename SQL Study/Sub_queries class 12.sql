drop database SQ;

create database SQ;

use SQ;

create table emp
(
emp_id int,
emp_name varchar(20),
emp_age int,
city varchar(20),
income int
);

create table emp2
(
emp_id int,
emp_name varchar(20),
emp_age int,
city varchar(20),
income int
);

insert into emp
values
(101,'Peter',32,'Newyork',200000),
(102,'Mark',32,'California',300000),
(103,'Doland',32,'Arizona',1000000),
(104,'Obama',32,'Florida',5000000),
(105,'Linklon',32,'Georgia',250000),
(106,'Kane',32,'Alaska',450000),
(107,'Adam',32,'California',5000000),
(108,'Macculam',32,'Florida',350000),
(109,'Brayan',32,'Alaska',400000),
(110,'Stephen',32,'Arizona',600000),
(111,'Alex',32,'California',70000);

insert into emp2
values
(101,'Peter',32,'Newyork',200000),
(103,'Doland',32,'Arizona',1000000),
(105,'Linklon',32,'Georgia',250000),
(107,'Adam',32,'California',5000000),
(109,'Brayan',32,'Alaska',400000),
(111,'Alex',32,'California',70000);

SELECT emp_name, city, income  
FROM emp WHERE income > (   
SELECT AVG(income) FROM emp);  

-- in statement (to use multiple statement in sub-queries) 
 
SELECT emp_id, emp_name, city, income FROM emp 
WHERE emp_id IN (SELECT emp_id FROM emp2);  

SELECT emp_id, emp_name, city, income FROM emp 
WHERE emp_id not IN (101,103,105,107);  

SELECT * FROM emp 
WHERE emp_id IN (SELECT emp_id FROM emp WHERE income > 350000);

SELECT emp_name, city, income FROM emp WHERE income = (SELECT MAX(income) FROM emp);




create table student
(
Stud_ID int,
`Name` varchar(20),
Email Varchar(30),
City varchar(20)
);

insert into student
values
(1,'Peter','peter@boffins.com','Texas'),
(2,'Suzi','suzi@boffins.com','California'),
(3,'Joseph','joseph@boffins.com','Alaska'),
(4,'Andrew','andrew@boffins.com','Los Angeles'),
(5,'Brayan','brayan@boffins.com','New York');

drop table student;

create table student2
(
Stud_ID int,
`Name` varchar(20),
Email Varchar(30),
City varchar(20)
);

insert into student2
values
(1,'Stephen','stephen@boffins.com','Texas'),
(2,'Joseph','joseph@boffins.com','Log Angeles'),
(3,'Peter','peter@boffins.com','California'),
(4,'David','david@boffins.com','New York'),
(5,'Maddy','maddy@boffins.com','Los Angeles');


SELECT Name, City FROM student  
WHERE City NOT IN (  
SELECT City FROM student2 WHERE City='Log Angeles');


create table emp3
(
emp_id int,
emp_name varchar(20),
emp_age int,
city varchar(20),
income int
);

insert into emp3
values
(101,'Peter',32,'Newyork',200000),
(102,'Mark',32,'California',300000),
(103,'Doland',32,'Arizona',1000000),
(104,'Obama',32,'Florida',5000000),
(105,'Linklon',32,'Georgia',600000),
(106,'Kane',32,'Alaska',450000),
(107,'Adam',32,'California',5000000),
(108,'Macculam',32,'Florida',350000),
(109,'Brayan',32,'Alaska',400000),
(110,'Stephen',32,'Arizona',600000),
(111,'Alex',32,'California',70000);

SELECT distinct(income) FROM sq.emp3 order by income desc;

SELECT distinct(income) FROM sq.emp3 order by income desc limit 2,1;

SELECT * FROM emp3 
WHERE income = (SELECT distinct(income) FROM sq.emp3 order by income desc limit 2,1); 
