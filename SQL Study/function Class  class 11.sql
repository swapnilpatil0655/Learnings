drop  database func_study;

create database func_study;

use func_study;

CREATE FUNCTION function_name (parameter datatype ,parameter datatype)
RETURNS return_datatype  
BEGIN  
Declaration_section  
Executable_section  
END;  

DELIMITER $$
create function add_to_col(a INT)
returns INT 
DETERMINISTIC
BEGIN 
	DECLARE b int ;
	set b = a + 10 ;
	return b ;
end $$

select add_to_col(20);

select * from sales

DELIMITER $$
create function add_to_col2(a INT)
returns decimal(10,2) 
DETERMINISTIC
BEGIN 
	DECLARE b int ;
	set b = a + 10 ;
	return b ;
end $$

select add_to_col2(20);

USE sales
select quantity, add_to_col(quantity) from sales_data_final;

DELIMITER $$
create function final_profits(profit DECIMAL(38, 6) , discount DECIMAL(38, 6))
returns DECIMAL(38, 6)
Deterministic
Begin 
Declare final_profit DECIMAL(38, 6) ;
set final_profit = profit - discount ;
return final_profit;
end $$

select profit, discount  , final_profits(profit, discount) from sales_data_final ; 

drop function final_profits;


alter table sales_data_final
add column fp decimal(38,6) after profit;

update sales_data_final
set fp  = final_profits_real(profit, discount,sales);

set sql_safe_updates = 0;

DELIMITER $$
create function final_profits_real(profit decimal(38,6) , discount decimal(38,6) , sales decimal(38,6) )
returns decimal(38,6)
Deterministic
Begin 
Declare final_profit decimal(38,6) ;
set final_profit = profit - sales * discount ;
return final_profit;
end $$


select profit, discount  ,sales ,  final_profits_real(profit, discount,sales) from sales_data_final; 


DELIMITER &&
create function int_to_str (a int)
returns varchar(30)
DETERMINISTIC
begin
declare b int;
set b = a;
return b;
end&&

select int_to_str(45) 

select * from sales1 

select quantity, int_to_str(quantity) from sales1 ; 

select max(sales) , min(sales) from sales1 


1  - 100 - super affordable product 
100-300 - affordable 
300 - 600 - moderate price 
600 + - expensive 


DELIMITER &&
create function mark_sales(sales int ) 
returns varchar(30)
DETERMINISTIC
begin 
declare flag_sales varchar(30); 
if sales  <= 100  then 
	set flag_sales = "super affordable product" ;
elseif sales > 100 and sales < 300 then 
	set flag_sales = "affordable" ;
elseif sales >300 and sales < 600 then 
	set flag_sales = "moderate price" ;
else 
	set flag_sales = "expensive" ;
end if ;
return flag_sales;
end &&


select mark_sales(100)

select sales , mark_sales(sales ) from practice.sales;



create table loop_table(val int);


Delimiter $$
create procedure insert_data()
Begin
set @var  = 10 ;
generate_data : loop
insert into loop_table values (@var);
set @var = @var + 1  ;
if @var  = 100 then 
	leave generate_data;
end if ;
end loop generate_data;
End $$

call insert_data()

select * from loop_table



Task 
	1 . Create a loop for a table to insert a record into a tale for two columns 
    in first coumn you have to inset a data ranging from 1 to 100 and 
    in second column you hvae to inset a square of the first column 
    2 . create a UDF which will be able to check a total number of records avaible in your table 
    3 . create a procedure to find out  5th highest profit in your sales table