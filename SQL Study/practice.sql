create database test;
use test;
show databases ;
create table test
(
Sr_no int,
Name varchar(20)
);

select * from test;

insert into test
value
(1,'swapnil'),
(2,'Nikhil');

insert into test
(Sr_no)
value
(3);

insert into test
(Name)
value
("Snehal");

insert into test 
set Sr_no =3
where 'Name' = "Snehal";

select * from test where Sr_no<3;
delete from test where Name ="Snehal";
set sql_safe_updates=0;
delete Sr_no from test where Name ="Nikhil"

-- day 2--

rename table test to est; 
use test
alter table est
add column age int;
set sql_safe_updates=0;
alter table est
drop column age

select * from est;
alter table est
add column age int;

alter table est
modify column age varchar(20);

alter table est 
modify column age int;

alter table est
rename column age to ph_no ;

alter table est
add column age int after ph_no ;

update est
set ph_no = 2564
where Name ="Swapnil";

update est
set age =27
where Name ="Swapnil"