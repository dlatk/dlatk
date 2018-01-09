.. _fwflag_create_rand_sample:
======================
--create_random_sample
======================
Switch
======

--create_random_sample percentage [random_seed]

Description
===========

Creates a new table with a random subset of rows from the table specified by :doc:`../fwinterface/fwflag_t`. New table has *_rand* appended to the table name.

You must specify the percentage of rows to keep. Optionally you can give a random seed, which is 42 by default. 

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`
* :doc:`fwflag_t`

Example Commands
================

Example 1: create the table *msgs_rand* that contains a random 10% of the rows in *msgs*:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs --create_random_sample .10

Example 2: create the table *blog_outcomes_rand* that contains a random 50% of the rows in *blog_outcomes* with the random seed 567:

.. code-block:: bash

	
	dlatkInterface.py -d dla_tutorial -t blog_outcomes --create_random_sample .50 567
