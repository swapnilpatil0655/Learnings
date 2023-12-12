create database proud ;


use proud;

Create table student
-- primary key is not null and unique-- 
(
id int primary key,
`name` varchar(20) Not Null, 
city varchar(20) default 'Nagpur',
age int check (age>=20),
ph bigint unique
);

insert into student
(id,name,city,age,ph)
values
(108,'Pranay','Delhi',20,123456785);

select * from student;
drop tables cource;
create table cource 
( 
sr int auto_increment primary key,
id int,
cource_name varchar(10),
foreign key (id) references student(id));

insert into cource
(id,cource_name)
values
(105,"DA"),
(106,"Java"),
(107,"DA"),
(108,"Python"),
(200,"Java");



delete from table student where id = 105;