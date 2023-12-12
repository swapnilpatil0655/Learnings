use practice1;
select * from bank_details;


DELIMITER &&
create procedure hi()
begin
	select * from bank_details;
end &&
call hi();

DELIMITER &&
create procedure hello()
begin
	select * from bank_details;
end &&

call hello()

DELIMITER &&
create procedure min_bal()
begin
	select * from bank_details order by balance limit 1;
end &&

call min_bal


Delimiter &&
create procedure avg_bal_jobrole(in var varchar(20))
begin
	select avg(balance) from bank_details where job=var;
end &&

call avg_bal_jobrole('management');
call avg_bal_jobrole('admin.');

select avg(balance) from bank_details where job='admin.';

delimiter &&
create procedure sel_edu_job(in var1 varchar(20), in var2 varchar (20))
begin
	SELECT * FROM bank_details where education = var1 and job = var2;
end &&
call sel_edu_job('primary','admin.');

create view bank_view as select age,job,balance,housing,marital,loan from bank_details;
select * from bank_details;
