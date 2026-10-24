.. _fwflag_anscombe:
==========
--anscombe
==========
Switch
======

--anscombe

Description
===========

A variance stabilizing transformation of the group norm

Argument and Default Value
==========================

Normalizes the group norm frequency information using the following formula:

Typically used to transform a random variable with a Poisson distribution to one with an approximately standard Gaussian.

Details
=======

Normalizes the group norm frequency information using the following formula:

Typically used to transform a random variable with a Poisson distribution to one with an approximately standard Gaussian. 


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

The transformation switches are used during feature extraction and therefore need at least one feature extraction command: 

* :doc:`fwflag_add_ngrams`, :doc:`fwflag_add_lex_table`, etc.

Example Commands
================

.. code-block:: bash


	# Creates the table: feat$cat_LEX_TABLE$TABLE$GROUP_BY_FIELD$16to8 
	dlatkInterface.py -d DATABSE -t TABLE -c GROUP_BY_FIELD --add_lex_table -l LEX_TABLE  --anscombe
	
	# Creates the table: feat$cat_met_a30_2000_cp_w$primals_new$dp_id$16to8
	dlatkInterface.py -d primals -t primals_new -c dp_id --group_freq_thresh 40000 --add_lex_table -l met_a30_2000_cp --weighted_lex --anscombe