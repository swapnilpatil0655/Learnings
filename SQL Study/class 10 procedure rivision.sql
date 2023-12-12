SELECT * FROM sales.sales_data_final;
use sales;

delimiter &&
create procedure order_ship_mode(in var varchar(20))
begin
	select count(*) from sales.sales_data_final where ship_mode = var;
end &&

call order_ship_mode('Second Class');

select distinct(ship_mode) from sales_data_final;

delimiter &&
create procedure country_wise_sales(in var varchar (30))
begin
	select sum(sales) from sales.sales_data_final where country = var;
end &&

call country_wise_sales('Brazil');

select sum(sales) from sales_data_final where country ='Brazil';
select country, sum(sales) from sales_data_final where country ='Brazil';
