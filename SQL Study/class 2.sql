use practice

rename table student to stu

-- Adding a column --
alter table stu
add column ph_no bigint  

alter table stu
add column ph bigint

select * from stu

#-------- Dropping a column ----------

alter table stu
drop column ph_no; 

select * from stu

#---------- Changing data type of column -------------
alter table stu
modify age varchar(5);

#----------- Rwnaming a column ------------------
alter table stu
rename column ph to phone_number;

select * from stu

alter table stu
add column id varchar(50) after Name;

select * from stu
alter table stu

update stu
set id="1"
where Name="Ankit"

-- for turn off the safe mode --
set sql_safe_updates =0;  

update stu
set id=2
where Name="Bhushan"

select * from stu

update stu
set id=3
where Name="Shreya"

update stu 
set id=4
where Name="Swapnil"

select * from stu

update stu
set City="Nagpur"
where id=4