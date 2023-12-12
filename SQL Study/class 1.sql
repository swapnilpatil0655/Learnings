create database practice;
drop database practice;
create database practice;
use practice;
-- for create comment use shortcut keys ctrl+?/ 
 show databases;
 use practice;
 create table Student
 (
 `Name` varchar(20),
 `Age` int,
 `City` varchar(15)
 );
 insert into Student
 value
 ("Ankit",28,"Mumbai"),
 ("Bhushan",29,"pune"),
 ("Shreya",24,"Agra"); 
 select * from student
 
 insert into Student
 (Name,Age)
 value
 ("Swapnil",22);
select * from student

insert into Student
(City,Name)
value
("Jaipur","Rajat");

select Name,Age from student;

where Statement
select * from Student where age>25;


-- for turn off the safe mode --
set sql_safe_updates =0;  

delete from Student where name ="Rajat";
select * from student
