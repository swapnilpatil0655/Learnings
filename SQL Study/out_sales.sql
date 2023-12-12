create database sales;
use sales;
CREATE TABLE sales_data_final (
	order_id VARCHAR(15) NOT NULL, 
	order_date varchar(20) NOT NULL, 
	ship_date varchar(20) NOT NULL, 
	ship_mode VARCHAR(14) NOT NULL, 
	customer_name VARCHAR(22) NOT NULL, 
	segment VARCHAR(11) NOT NULL, 
	state VARCHAR(36) NOT NULL, 
	country VARCHAR(32) NOT NULL, 
	market VARCHAR(6) NOT NULL, 
	region VARCHAR(14) NOT NULL, 
	product_id VARCHAR(16) NOT NULL, 
	category VARCHAR(15) NOT NULL, 
	sub_category VARCHAR(11) NOT NULL, 
	product_name VARCHAR(127) NOT NULL, 
	sales DECIMAL(38, 0) NOT NULL, 
	quantity DECIMAL(38, 0) NOT NULL, 
	discount DECIMAL(38, 3) NOT NULL, 
	profit DECIMAL(38, 5) NOT NULL, 
	shipping_cost DECIMAL(38, 2) NOT NULL, 
	order_priority VARCHAR(8) NOT NULL, 
	year DECIMAL(38, 0) NOT NULL
);

load data infile "D:/SQL Study/sales_data_final.csv"
into table sales_data_final
fields terminated by','
enclosed by'"'
lines terminated by "\n"
ignore 1 rows;

set session sql_mode="";

alter table sales_data_final
add column order_data_new date after order_date;

----------- for convert string to date format ----------------
select str_to_date(order_date,'%m/%d/%Y') from sales_data_final;

update sales_data_final
set order_data_new = str_to_date(order_date,'%m/%d/%Y');

-------------- for drop a column ----------
alter table sales_data_final
drop column order_data_new;
---------- -----------

alter table sales_data_final
add column order_date_new date after order_date;

set order_date_new = str_to_date(order_date,'%m/%d/%Y') from sales_data_final;

update sales_data_final
set order_date_new = str_to_date(order_date,'%m/%d/%Y');

alter table sales_data_final
add column ship_date_new date after ship_date;

select str_to_date(ship_date,'%m/%d/%Y') from sales_data_final;

update sales_data_final
set ship_date_new =  str_to_date(ship_date,'%m/%d/%Y');

alter table sales_data_final
drop column order_date;

alter table sales_data_final
drop column ship_date;



