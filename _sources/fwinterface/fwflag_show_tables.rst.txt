.. _fwflag_show_tables:
=============
--show_tables
=============
Switch
======

--show_tables [like]

Description
===========

See all available non-feature tables in a given database

SQL style wildcards ('%') will work with the optional *like* argument

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`

Example Commands
================

Show all non-feature tables:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial 

	Found 5 available tables
	blog_outcomes
	blog_outcomes_rand
	dummy_table
	msgs
	msgs_rand

Show all non-feature tables that begin with *blog*:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial  --show_tables 'blog%'  

	Found 2 available tables
	blog_outcomes
	blog_outcomes_rand

