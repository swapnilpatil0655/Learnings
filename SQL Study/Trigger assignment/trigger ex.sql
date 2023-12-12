CREATE TABLE insert_record (
	`PRODUCT ID` VARCHAR(5) NOT NULL, 
	`QUANTITY` DECIMAL(38, 0) NOT NULL, 
	`SALE TYPE` VARCHAR(12) NOT NULL, 
	`PAYMENT MODE` VARCHAR(6) NOT NULL, 
	`DISCOUNT %%` BOOL NOT NULL,
    `TIME` TIME ,
    `USER` VARCHAR(20)
);
 
DELIMITER %%
CREATE trigger after_insert
after insert on product_sales for each row
begin
	insert into insert_record
    values
    (new.`PRODUCT ID` , new.`QUANTITY`,new.`SALE TYPE`,new.`PAYMENT MODE`, NEW.`DISCOUNT %%`,curtime(),user());
END %%

---------------------------------------------------------------------------------
-- delete record --------------------
 CREATE TABLE delete_record (
	`PRODUCT ID` VARCHAR(5) NOT NULL, 
	`QUANTITY` DECIMAL(38, 0) NOT NULL, 
	`SALE TYPE` VARCHAR(12) NOT NULL, 
	`PAYMENT MODE` VARCHAR(6) NOT NULL, 
	`DISCOUNT %%` BOOL NOT NULL,
    `TIME` TIME ,
    `USER` VARCHAR(20)
);

DELIMITER %%
CREATE trigger after_delete
after delete on product_sales for each row
begin
	insert into delete_record
    values
    (old.`PRODUCT ID` , old.`QUANTITY`,old.`SALE TYPE`,old.`PAYMENT MODE`,old.`DISCOUNT %%`,curtime(),user());
END %%