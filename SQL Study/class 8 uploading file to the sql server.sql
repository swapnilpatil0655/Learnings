use practice;
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
lines terminated by'\n'
ignore 1 rows;

SELECT * FROM practice.sales_data_final;

-- This syntx is used for remove the , from the database-- 
set session sql_mode=""; 

set sql_updates=0;
set global max_allowed_packded =1048576;
