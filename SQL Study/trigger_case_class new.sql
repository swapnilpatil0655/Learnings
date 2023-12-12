DROP database trigger_study;

create database trigger_study1;

use trigger_study1;

CREATE TABLE employee(  
    `name` varchar(45) NOT NULL,    
    occupation varchar(35) NOT NULL,    
    working_date date,  
    working_hours varchar(10)  
); 

INSERT INTO employee VALUES    
('Robin', 'Scientist', '2020-10-04', 12),  
('Warner', 'Engineer', '2020-10-04', 10),  
('Peter', 'Actor', '2020-10-04', 13),  
('Marco', 'Doctor', '2020-10-04', 14),  
('Brayden', 'Teacher', '2020-10-04', 12),  
('Antonio', 'Business', '2020-10-04', 11);  

select * from employee

------------------------------------- Before Insert-----------------------------------------------------
DELIMITER //  
Create Trigger before_insert_empworkinghours   
BEFORE INSERT ON employee FOR EACH ROW  
BEGIN  
IF NEW.working_hours < 0 THEN SET NEW.working_hours = 0;  
END IF;  
END //
delimiter ; 

drop trigger before_insert_empworkinghours

INSERT INTO employee VALUES    
('Markus', 'Former', '2020-10-08', 14),
('Alexander', 'Actor', '2020-10-012', -13);

----------------------------------------------------------------------------------------------------------

-------------------------------------------- AFTER INSERT ------------------------------------------------
CREATE TABLE student_info (  
  stud_id int NOT NULL,  
  stud_code varchar(15) DEFAULT NULL,  
  stud_name varchar(35) DEFAULT NULL,  
  `subject` varchar(25) DEFAULT NULL,  
  marks int DEFAULT NULL,  
  phone varchar(15) DEFAULT NULL,  
  PRIMARY KEY (stud_id)  
) ;

CREATE TABLE student_log (  
  stud_id int NOT NULL,  
  stud_code varchar(15) DEFAULT NULL,  
  stud_name varchar(35) DEFAULT NULL,  
  subject varchar(25) DEFAULT NULL,  
  marks int DEFAULT NULL,  
  phone varchar(15) DEFAULT NULL,  
  Lasinserted Time,  
  PRIMARY KEY (stud_id)  
); 

DELIMITER //  
Create Trigger after_insert_details  
AFTER INSERT ON student_info FOR EACH ROW  
BEGIN  
INSERT INTO student_log VALUES (new.stud_id, new.stud_code,   
new.stud_name, new.`subject`, new.marks, new.phone, CURTIME());  
END //

drop trigger after_insert_details;
 
select curtime()
select now()
INSERT INTO student_info VALUES   
(01, 101, 'Armin', 'Maths', 82, '513548544'),
(02, 102, 'Jojo', 'Bio', 65, '4211589654'),
(03, 103, 'Goku', 'History', 96, '458859671'); 
select * from student_log;
INSERT INTO student_info VALUES   
(05, 105, 'Kira', 'Sci', 89, '985475896');
-------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------- BEFORE UPDATE -----------------------------------------------------

DELIMITER $$   
CREATE TRIGGER trigger_name 
BEFORE UPDATE ON table_name FOR EACH ROW  
BEGIN  
   variable declarations  
   trigger code  
END$$  
DELIMITER ;  

CREATE TABLE sales_info 
(  
id INT AUTO_INCREMENT,  
product VARCHAR(100) NOT NULL,  
quantity INT NOT NULL DEFAULT 0,  
fiscalYear SMALLINT NOT NULL,  
CHECK(fiscalYear BETWEEN 2000 and 2050),  
CHECK (quantity >=0),  
UNIQUE(product, fiscalYear),  
PRIMARY KEY(id)  
); 

INSERT INTO sales_info(product, quantity, fiscalYear)  
VALUES  
('2003 Maruti Suzuki',110, 2020),  
('2015 Avenger', 120,2020),  
('2018 Honda Shine', 150,2020),  
('2014 Apache', 150,2020);

DELIMITER $$  
CREATE TRIGGER before_update_salesInfo  
BEFORE UPDATE  
ON sales_info FOR EACH ROW  
BEGIN  
    IF new.quantity > old.quantity * 2 THEN  
    set  new.quantity = old.quantity;
    END IF;  
END $$  
DELIMITER ; 

drop trigger before_update_salesInfo;

DELIMITER $$  
CREATE TRIGGER before_update_salesInfo  
BEFORE UPDATE  
ON sales_info FOR EACH ROW  
BEGIN  
    DECLARE error_msg VARCHAR(255);  
    SET error_msg = ('The new quantity cannot be greater than 2 times the current quantity');  
    IF new.quantity > old.quantity * 2 THEN  
    SIGNAL SQLSTATE '45000'   
    SET MESSAGE_TEXT = error_msg;  
    END IF;  
END $$  
DELIMITER ; 

 UPDATE sales_info SET quantity = 125 WHERE id = 2;   
 
 UPDATE sales_info SET quantity = 600 WHERE id = 2;  
 
 ----------------------------------------------------------------------------------------------------
 
 ------------------------------------------- AFTER UPDATE  --------------------------------------
 
 CREATE TABLE stu(    
    id int NOT NULL AUTO_INCREMENT,    
    name varchar(45) NOT NULL,    
    class int NOT NULL,    
    email_id varchar(65) NOT NULL,    
    PRIMARY KEY (id)    
);  

INSERT INTO stu (name, class, email_id)     
VALUES ('Stephen', 6, 'stephen@javatpoint.com'),   
('Bob', 7, 'bob@javatpoint.com'),   
('Steven', 8, 'steven@javatpoint.com'),   
('Alexandar', 7, 'alexandar@javatpoint.com'); 

CREATE TABLE stu_log(    
    user varchar(45) NOT NULL,    
    descreptions varchar(65) NOT NULL  
);  

DELIMITER $$  
CREATE TRIGGER after_update_studentsInfo  
AFTER UPDATE  
ON stu FOR EACH ROW  
BEGIN  
    INSERT into stu_log VALUES (user(),   
    CONCAT('Update Student Record ', OLD.name, ' Previous Class :',  
    OLD.class, ' Present Class ', NEW.class));  
END $$    
DELIMITER ;  

UPDATE stu SET class = class + 1;  
set sql_safe_updates =0
----------------------------------------------------------------------------------------------------------

--------------------------------------------------- BEFORE DELETE ----------------------------------------

CREATE TABLE salaries (  
    emp_num INT PRIMARY KEY,  
    valid_from DATE NOT NULL,  
    amount DEC(8 , 2 ) NOT NULL DEFAULT 0  
); 

INSERT INTO salaries (emp_num, valid_from, amount)  
VALUES  
    (102, '2020-01-10', 45000),  
    (103, '2020-01-10', 65000),  
    (105, '2020-01-10', 55000),  
    (107, '2020-01-10', 70000),  
    (109, '2020-01-10', 40000);  
    
CREATE TABLE salary_log (  
    id INT PRIMARY KEY AUTO_INCREMENT,  
    emp_num INT,  
    valid_from DATE NOT NULL,  
    amount DEC(18 , 2 ) NOT NULL DEFAULT 0,  
    deleted_time TIMESTAMP DEFAULT NOW()  
);  

DELIMITER $$  
  
CREATE TRIGGER before_delete_salaries  
BEFORE DELETE  
ON salaries FOR EACH ROW  
BEGIN  
    INSERT INTO salary_log (emp_num, valid_from, amount)  
    VALUES(OLD.emp_num, OLD.valid_from, OLD.amount);  
END$$   
  
DELIMITER ;  

DELETE FROM salaries WHERE emp_num = 102;  

SELECT * FROM salary_log;  

-------------------------------------------------------------------------------------------------

-------------------------------------------------- AFTER DELETE  -----------------------------------------------

CREATE TABLE salaries (  
    emp_num INT PRIMARY KEY,  
    valid_from DATE NOT NULL,  
    amount DEC(8 , 2 ) NOT NULL DEFAULT 0  
);  

delete from salaries;

set sql_safe_updates = 0;

INSERT INTO salaries (emp_num, valid_from, amount)  
VALUES  
    (102, '2020-01-10', 45000),  
    (103, '2020-01-10', 65000),  
    (105, '2020-01-10', 55000),  
    (107, '2020-01-10', 70000),  
    (109, '2020-01-10', 40000);  
    
CREATE TABLE total_salary_budget(  
    total_budget DECIMAL(10,2) NOT NULL  
);  

INSERT INTO total_salary_budget
SELECT SUM(amount) FROM salaries;  

DELIMITER $$  
CREATE TRIGGER after_delete_salaries  
AFTER DELETE  
ON salaries FOR EACH ROW  
BEGIN  
   UPDATE total_salary_budget SET total_budget = total_budget - old.amount;  
END$$   
DELIMITER ;  

DELETE FROM salaries WHERE emp_num = 105;  

SELECT * FROM total_salary_budget;  

DELETE FROM salaries;  

---------------------------------------------------------------------------------------------------------------------

