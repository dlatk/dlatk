.. _fwflag_desc_tables:
=================
--describe_tables
=================
Switch
======

--describe_tables 

Description
===========

Describe MySQL tables. Will print the description of the tables specified by :doc:`fwflag_t`, :doc:`fwflag_f` and :doc:`fwflag_outcome_table` if any of those flags are present. Also takes optional positional arguments and prints the description of those tables as well. 

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

	dlatkInterface.py -d dla_tutorial -t msgs --describe_tables

	SQL QUERY: DESCRIBE msgs
	                    Field                     Type      Null       Key   Default          Extra
	               message_id                  int(11)        NO       PRI           auto_increment
	                  user_id         int(10) unsigned       YES       MUL                         
	                     date              varchar(64)       YES                                   
	             created_time                 datetime       YES       MUL                         
	                  message                     text       YES                                   

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs --describe_tables blog_outcomes 

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

