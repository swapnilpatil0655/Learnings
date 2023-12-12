use practice;
-- Q.1) Write a procedure to find out the maximum value of Balance ? -- (table:- bank_details)

delimiter &&
create procedure max_value()
begin
	select max(balance) from bank_details;
end &&

call max_value()

-- Q.2) Write procedure named renter which will return record of customers who are renting ?

delimiter //
create procedure renter()
begin
	select * from bank_details where housing = 'no';
end//

call renter()

-- Q.3) Write a procedure loan_known taker which will return record for employee word has the loan ?
delimiter //
create procedure loan_known()
begin
	select * from bank_details where loan ='yes';
end//

call loan_known()

-- Q.4) Write a procedure named single which will show the record for each customer who are unmarried?
delimiter // 
create procedure ssingle()
begin
	select * from bank_details where marital ='single';
end//

call ssingle()

-- Q.5) Writer procedure named sel_edu which will take education as an argument and return record of customer of given education category?--  
delimiter //
create procedure sel_edu(in var varchar(20))
begin
	select * from bank_details where education = var;
end//

call sel_edu('secondary')

-- Q.6) procedure which will take age and marital as input and show record of customer which satisfy those age and merital condition?--
 delimiter //
 create procedure record_age_marital()
 begin
	select * from bank_details where age = 44 and marital='single';
end//

call record_age_marital()
--------------------------------------------------------------------------------------------------------
delimiter //
 create procedure rrecord_age_marital(in num int ,in var varchar(20))
 begin
	select * from bank_details where age = num and marital= var;
end//

call rrecord_age_marital(44,'single')
---------------------------------------------------------------------------------------------------------

-- Q.7) Write a procedure named sel_payment_mode which will take payment as an argument and return record for product satisfy that condition? (table:- ProductSales) -- 
delimiter // 
create procedure ssel_payment_mode (in var varchar(20))
begin
	select * from productsales where PAYMENT_MODE = var;
end//

call ssel_payment_mode ('Cash')

-- Q.8) Write a procedure which will take sale type as an input and return total quantity sold for that sale type? --
delimiter //
create procedure total_quantity_by_type(in var varchar(20))
begin
	select sum(QUANTITY) from productsales where SALE_TYPE= var;
end //

call total_quantity_by_type('Online')

-- Q.9) Write a procedure to show total quantity of each product and order it according to product id? -- 
delimiter //
create procedure ttotal_quantity_by_product()
begin
	select PRODUCT_ID, sum(QUANTITY) from productsales group by PRODUCT_ID order by (PRODUCT_ID);  
end //

call ttotal_quantity_by_product()

-- Q.10) Write a procedure which will take product ID and payment mode as an argument and return record which satisfies both conditions? --
delimiter //
create procedure pproduct_id_n_payment_mode(in var int,in pmod varchar(20))
begin
	select * from productsales where PRODUCT_ID = var and PAYMENT_MODE = pmod;
end //

call product_id_n_payment_mode(128 , 'Online')
