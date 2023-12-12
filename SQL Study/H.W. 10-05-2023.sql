use practice;
select * from bank_details;
-- Q.1
select * from bank_details where loan="yes" ;

select * from bank_details where education="primary" or job="unknown" ;
select * from bank_details where housing="no" and marital="single" ;
select age,count(age) from bank_details group by age ;
select * from bank_details where job="unknown";
select age, avg (balance) from bank_details where job="unknown" group by age;

update bank_details 
set housing="no"
where age='60';




delete from bank_details where age="24";
set sql_safe_update=0;


--

create database practice1;
use practice1;
select * from bank_details;
select * from bank_details where job="unknown";

update bank_details 
set housing='no'
where age='60';
select * from bank_details where age =60;

set sql_safe_updates=0;

delete from bank_details where age="24";

 
