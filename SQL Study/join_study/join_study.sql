create database join_study;
use join_study;

create table emp
(
emp_id int,
emp_name varchar(20),
emp_age int,
city_id int
);

insert into emp
values
(101,'Peter',32,1),
(102,'Mark',32,2),
(103,'Doland',36,4),
(104,'Obama',32,3),
(105,'Linklon',32,5),
(106,'Kane',32,1),
(107,'Adam',32,2),
(108,'Macculam',32,2),
(109,'Brayan',32,3),
(110,'Stephen',32,4),
(111,'Alex',32,5);

create table city_tab
(
sr_no int,
city_name varchar(20)
);

insert into city_tab
values
(1,"Newyork"),
(2,"Arizona"),
(3,"California");

select *
from emp
inner join city_tab on emp.city_id = city_tab.sr_no;

select emp.emp_id,emp.emp_name,emp.emp_age,city_tab.city_name
from emp
cross join city_tab;

select emp.emp_id,emp.emp_name,emp.emp_age,city_tab.city_name
from emp
left join city_tab on emp.city_id = city_tab.sr_no;

-------------------- Alias (fake name for table id ---------------------
select e.emp_id, e.emp_name,e.emp_age,c.city_name
from emp e
left join city_tab c on e.city_id = c.sr_no
------------------------------------------------------------------------

select emp.emp_id,emp.emp_name,emp.emp_age,emp.city_id,city_tab.sr_no,city_tab.city_name
from emp
right join city_tab on emp.city_id = city_tab.sr_no;

