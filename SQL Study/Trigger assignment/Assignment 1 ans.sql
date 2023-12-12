-------------------- Q.1 -------------------------
delimiter //
create trigger afterinsert_car_sales
after insert on car_sales for each row
begin
insert into carsales_log values (new.Order_Number,new.Order_Date,new.Unit_Price,new.Total_Revenue,new.Total_Revenue_cost);
end //

-------------------- Q.2----------------------

delimiter //
create trigger convert_into_0
before insert on citystore for each row
begin
if new.store_size < 0 then
set new.store_size = 0;
end if ;
end //
delimiter ;
 
-------------------- Q.3 ------------

delimiter //
create trigger new_record
after insert on Customer for each row 
begin
insert into Cus_log value(new.customer_id,new.Name,new.Surname,new.Gender,new.Age,new.Region,new.Job_classification,new.Date_joined,new.Balance);
end//
delimiter ;

----------------- Q.4 --------------------

delimiter //
create trigger negative_update_0
after insert on Customer for each row
begin
if new.balance < 0 then set new.balance = 0;
end if;
end //
delimiter ;

------------------ -- Q.5 -----------------
delimiter //
create trigger `name of trigger`
before/after (insert/update/delete) on `TABLE_NAME` for each row 
begin
formula / statement
end //
delimiter ;
