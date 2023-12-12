-------- Agrigation function ----------------- 
use practice;
select sum(balance) from bank_details;
select sum(age) from bank_details;
select avg(age) from bank_details;
select min(balance) from bank_details;
select max(balance) from bank_details;
select count(balance) from bank_details;
select count(*) from bank_details;

------------ Distinct value (select every individual value)
select distinct(age) from bank_details;
select count(distinct(age)) from bank_details;

select age, avg(balance) from bank_details group by age ;
select age, avg(balance) from bank_details group by age order by age ;
select age, avg(balance) from bank_details group by age order by age desc;

select * from bank_details order by age limit 5;
select * from bank_details order by age desc limit 5;
select * from bank_details order by age limit 1,2;