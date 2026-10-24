.. _fwflag_boolean:
=========
--boolean
=========
Switch
======

--boolean

Description
===========

A boolean transformation of the group norm frequency information.

Argument and Default Value
==========================

Normalizes the group norm frequency information using the following formula:

Details
=======

Normalizes the group norm frequency information using the following formula:



Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

The transformation switches are used during feature extraction and therefore need at least one feature extraction command: 

* :doc:`fwflag_add_ngrams`, :doc:`fwflag_add_lex_table`, etc.


Example Commands
================

.. code-block:: bash


	# Creates the table: feat$cat_LEX_TABLE$TABLE$GROUP_BY_FIELD$16to1 
	dlatkInterface.py -d DATABSE -t TABLE -c GROUP_BY_FIELD --add_lex_table -l LEX_TABLE --boolean` 
	
	# Creates the table: feat$cat_met_a30_2000_cp_w$primals_new$dp_id$16to1
	dlatkInterface.py -d primals -t primals_new -c dp_id --group_freq_thresh 40000 --add_lex_table -l met_a30_2000_cp --weighted_lex --boolean 
