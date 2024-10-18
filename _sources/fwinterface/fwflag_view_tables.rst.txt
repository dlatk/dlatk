.. _fwflag_view_tables:
=============
--view_tables
=============
Switch
======

--view_tables 

Description
===========

Show data from MySQL tables. Will print the first five rows of the tables specified by :doc:`fwflag_t`, :doc:`fwflag_f` and :doc:`fwflag_outcome_table` if any of those flags are present. Also takes optional positional arguments and prints the description of those tables as well. 

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`

Optional Switches: 

* :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`

Example Commands
================

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




