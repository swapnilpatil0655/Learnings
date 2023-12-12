use trigger_study;

CREATE TABLE `Product_Sales` (
	`PRODUCT ID` VARCHAR(5) NOT NULL, 
	`QUANTITY` DECIMAL(38, 0) NOT NULL, 
	`SALE TYPE` VARCHAR(12) NOT NULL, 
	`PAYMENT MODE` VARCHAR(6) NOT NULL, 
	`DISCOUNT %%` BOOL NOT NULL
);

load data infile "D:/SQL Study/Trigger assignment/Product_Sales.csv"
INTO table Product_Sales
fields terminated by','
enclosed by '"'
lines terminated by "\n"
ignore 1 rows




CREATE TABLE insert_record (
	`PRODUCT ID` VARCHAR(5) NOT NULL, 
	`QUANTITY` DECIMAL(38, 0) NOT NULL, 
	`SALE TYPE` VARCHAR(12) NOT NULL, 
	`PAYMENT MODE` VARCHAR(6) NOT NULL, 
	`DISCOUNT %%` BOOL NOT NULL,
    `TIME` TIME ,
    `USER` VARCHAR(20)
);

