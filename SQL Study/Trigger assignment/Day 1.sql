#-------------------------- Creating Database -----------------------------
		-- syntx: Create database database_name
Create Database Boffins;
#--------------------------------------------------------------------------

#---------------------------Using a Database-------------------------------
		-- syntx: Use database_name
Use Boffins;
#--------------------------------------------------------------------------

#--------------------------Dropping a Table -------------------------------

		-- syntx: drop database database_name
drop database Boffins;
#--------------------------------------------------------------------------

#-------------------------Creating a table---------------------------------
		-- syntx: create table table_name
		-- 	      (
		--         col1 datatype,
		--         col2 datatype,
		--           |       |
		--         coln datatype
		-- 		  );
        
create table student
(
id int,
name varchar(20),
age int
);
#----------------------------------------------------------------------------