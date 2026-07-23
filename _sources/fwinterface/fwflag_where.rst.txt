.. _fwflag_where:
=======
--where
=======
Switch
======

--where VALUE

Description
===========

Filter groups based on sql call.

Argument and Default Value
==========================

Argument is sql-lik` query with column_name, operator and value. The column_name must be an actual column in :doc:`fwflag_outcome_table`. Do NOT include the word "where" after inside your. There is no default value.

Works with correlation, prediction and classification.

Examples:

* --where "age >= 30 and age < 80"
* --where "gender = 0"
* --where "upt > 100"

Other Switches
==============

Required Switches: 

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 
* :doc:`fwflag_outcome_table` 

Example Commands
================

.. code-block:: bash

	# Run LIWC over age and only consider males
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$cat_LIWC2007$msgs$user_id$16to16' --outcome_table blog_outcomes --outcomes age --where "gender = 1" --output_name xxx_output --correlate
