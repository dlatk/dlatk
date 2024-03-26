.. _fwflag_add_lex_table:
===============
--add_lex_table
===============
Switch
======

--add_lex_table [-l LEX]

Description
===========

Add a lexicon-based feature table

Argument and Default Value
==========================

--add_lex_table -l <LEX> (where <LEX> is the name of the table)

Details
=======

Given an existing N-gra,s table as your original feature table, this switch produces a new feature table where the features are now topics (as opposed to N:doc:`fwflag_grams`) based on the topics in the LEX table.

Unless :doc:`fwflag_f` is provided, the table feat$1gram$TABLE$GROUP_BY_FIELD$16to16 must be present.


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_l` 

Optional Switches:

* :doc:`fwflag_f` [alternative n-gram table; can be used in place of word table]
* :doc:`fwflag_weighted_lexicon` :doc:`fwflag_anscombe`, :doc:`fwflag_sqrt`, :doc:`fwflag_log`, :doc:`fwflag_boolean` (transformations, default is no transformation)

Example Commands
================

.. code-block:: python

	# General form
	# Creates the table: feat$cat_LEX_TABLE$TABLE$GROUP_BY_FIELD$16to# 
	# In the above # = (1, 2, 4, 8, 16) and depends on the transformation (boolean, log, sqrt, anscombe and none, respectively)
	./dlatkInterface.py -d DATABSE -t TABLE -c GROUP_BY_FIELD -add_lex_table -l LEX_TABLE

	# Creates the table: feat$cat_met_a30_2000_cp_w$primals_new$dp_id$16to16
	./dlatkInterface.py -d primals -t primals_new -c dp_id --group_freq_thresh 40000 --add_lex_table -l met_a30_2000_cp --weighted_lex