create database office_supplies;
use office_supplies;
create table supplies (
`order_date` date not null,
`region` varchar(20) ,
`rep` varchar (20),
`item` varchar(20),
`units` int,
`unit_price` numeric 
);

alter table supplies
modify `unit_price` int;

insert into supplies
values
("2014-06-04","East","Richard","Pen set",62,4.99);

update supplies
set unit_price = 4.99
where item = "Pen set";

SET SQL_SAFE_UPDATES = 0;

alter table supplies 
modify unit_price decimal(10,00);

update supplies
set unit_price = 4.99
where item ="Pen set";


insert into supplies
values 
("2014-