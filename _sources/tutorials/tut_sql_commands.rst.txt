.. _tut_sql_commands:
=================================
Using DLATK to view your SQL data
=================================

Since DLATK relies so heavily in MySQL we have created a few commands to help keep track of high level data. As with most dlatkInterface calls you will need the standard flags:

* :doc:`../fwinterface/fwflag_d`: the database we are using
* :doc:`../fwinterface/fwflag_t`: the table inside the database where our text lives (aka the message table)
* :doc:`../fwinterface/fwflag_c`: the table column we will be grouping the text by (aka group)

Viewing and Describing Tables
=============================

Show Feature Tables
-------------------

* :doc:`../fwinterface/fwflag_ls`

View all feature tables for a given message table and / or grouping

**Example 1**: See user level features for the message table *msgs*:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --show_feature_tables

	SQL QUERY: SHOW TABLES FROM dla_tutorial LIKE 'feat$%$msgs$user_id$%' 
	Found 4 available feature tables
	feat$1gram$msgs$user_id$16to1
	feat$1gram$msgs$user_id$16to16
	feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16
	feat$meta_1gram$msgs$user_id$16to1

**Example 2**: See features extracted at any level for the message table *msgs*:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c '%' --ls

	SQL QUERY: SHOW TABLES FROM dla_tutorial LIKE 'feat$%$msgs$%$%'
	Found 6 available feature tables
	feat$1gram$msgs$message_id$16to1
	feat$1gram$msgs$user_id$16to1
	feat$1gram$msgs$user_id$16to16
	feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16
	feat$meta_1gram$msgs$message_id$16to1
	feat$meta_1gram$msgs$user_id$16to1


Show Tables
-----------

* :doc:`../fwinterface/fwflag_show_tables`

Use this flag to view non-feature tables:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial --show_tables

	Found 5 available tables
	blog_outcomes
	blog_outcomes_rand
	dummy_table
	msgs
	msgs_rand

	dlatkInterface.py -d dla_tutorial  --show_tables 'blog%'  

	Found 2 available tables
	blog_outcomes
	blog_outcomes_rand


Describe Tables
---------------

* :doc:`../fwinterface/fwflag_desc_tables`

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs --describe_tables
	...
	SQL QUERY: DESCRIBE msgs
	                    Field                     Type      Null       Key   Default          Extra
	               message_id                  int(11)        NO       PRI           auto_increment
	                  user_id         int(10) unsigned       YES       MUL                         
	                     date              varchar(64)       YES                                   
	             created_time                 datetime       YES       MUL                         
	                  message                     text       YES                                   

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs --describe_tables blog_outcomes 
	...
	SQL QUERY: DESCRIBE msgs
	                    Field                     Type      Null       Key   Default          Extra
	               message_id                  int(11)        NO       PRI           auto_increment
	                  user_id         int(10) unsigned       YES       MUL                         
	                     date              varchar(64)       YES                                   
	             created_time                 datetime       YES       MUL                         
	                  message                     text       YES                                   
	SQL QUERY: DESCRIBE blog_outcomes
	                    Field                     Type      Null       Key   Default          Extra
	                  user_id                  int(11)        NO       PRI                         
	                   gender                   int(2)       YES                                   
	                      age          int(3) unsigned       YES                                   
	                     occu              varchar(32)       YES                                   
	                     sign              varchar(16)       YES                                   
	                is_indunk                   int(1)       YES                                   
	               is_student                   int(1)       YES                                   
	             is_education                   int(1)       YES                                   
	            is_technology                   int(1)       YES                                   

View Table Data
---------------

* :doc:`../fwinterface/fwflag_view_tables`

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs --view_tables
	...
	SQL QUERY: select column_name from information_schema.columns 
            where table_schema = 'dla_tutorial' and table_name='msgs'
	SQL QUERY: SELECT * FROM msgs LIMIT 5
     message_id        user_id           date   created_time        message
              1        3991108   31,July,2004 2004-07-31 00:    can you bel
              2        3991108   25,July,2004 2004-07-25 00:    miss su  us
              3        3991108   24,July,2004 2004-07-24 00:    i'm lookin 
              4        3991108   24,July,2004 2004-07-24 00:    what a time
              5        3991108 01,August,2004 2004-08-01 00:    i cannot be

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs --view_tables blog_outcomes 
	...
	SQL QUERY: select column_name from information_schema.columns 
            where table_schema = 'dla_tutorial' and table_name='msgs'
	SQL QUERY: SELECT * FROM msgs LIMIT 5
     message_id        user_id           date   created_time        message
              1        3991108   31,July,2004 2004-07-31 00:    can you bel
              2        3991108   25,July,2004 2004-07-25 00:    miss su  us
              3        3991108   24,July,2004 2004-07-24 00:    i'm lookin 
              4        3991108   24,July,2004 2004-07-24 00:    what a time
              5        3991108 01,August,2004 2004-08-01 00:    i cannot be

	SQL QUERY: select column_name from information_schema.columns 
	            where table_schema = 'dla_tutorial' and table_name='blog_outcomes'
	SQL QUERY: SELECT * FROM blog_outcomes LIMIT 5
	        user_id         gender     gender_cat            age           occu           sign      is_indunk     is_student   is_education  is_technology
	        3991108              1         female             17         indUnk            Leo              1              0              0              0
	        3417138              1         female             25 Communications         Taurus              0              0              0              0
	        3673414              0           male             14        Student        Scorpio              0              1              0              0
	        3361075              1         female             16        Student      Capricorn              0              1              0              0
	        4115327              1         female             14         indUnk          Libra              1              0              0              0


Creating tables
===============

These commands allow you to create random samples of your data

Random Sample
-------------

* :doc:`../fwinterface/fwflag_create_rand_sample`

Creates a new table with a random subset of rows from the table specified by :doc:`../fwinterface/fwflag_t`.

**Example 1**: create the table *msgs_rand* that contains a random 10% of the rows in *msgs*:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs --create_random_sample .10
	...
	SQL QUERY: DROP TABLE IF EXISTS msgs_rand
	SQL QUERY: CREATE TABLE msgs_rand LIKE msgs
	SQL QUERY: ALTER TABLE msgs_rand DISABLE KEYS
	SQL QUERY: INSERT INTO msgs_rand SELECT * FROM msgs where RAND(42) < 0.11000000000000001 LIMIT 3167
	SQL QUERY: ALTER TABLE msgs_rand ENABLE KEYS

**Example 2**: create the table *blog_outcomes_rand* that contains a random 50% of the rows in *blog_outcomes* with the random seed 567:

.. code-block:: bash

	
	dlatkInterface.py -d dla_tutorial -t blog_outcomes --create_random_sample .50 567
	...
	SQL QUERY: DROP TABLE IF EXISTS msgs_rand
	SQL QUERY: CREATE TABLE msgs_rand LIKE msgs
	SQL QUERY: ALTER TABLE msgs_rand DISABLE KEYS
	SQL QUERY: INSERT INTO msgs_rand SELECT * FROM msgs where RAND(567) < 0.55 LIMIT 15837
	SQL QUERY: ALTER TABLE msgs_rand ENABLE KEYS


