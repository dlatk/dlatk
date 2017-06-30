.. _fwflag_combine_feat_tables:
=====================
--combine_feat_tables
=====================
Switch
======

--combine_feat_tables NEW_FEAT_NAME

Description
===========

Combines multiple feature tables into one with a new name.

Argument and Default Value
==========================

name of the new feature

Details
=======

Takes all the feature tables listed after :doc:`fwflag_f` and combines them into one new feature table. The new table will have the argument as the feature name (i.e. feat$NEW_FEAT_NAME$...).
The new table will have an index on the group_id and feat columns


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f` 

Optional Switches:

* :doc:`fwflag_add_ngrams`
* :doc:`fwflag_add_ngrams_from_tokenized`
* :doc:`fwflag_feat_occ_filter`
* :doc:`fwflag_feat_colloc_filter` 

Example Commands
================

.. code-block:: bash


	# Simply combines two tables:
	# Creates: feat$1to2gram$messages$user_id$16to16
	dlatkInterface.py -d 2004blogs -t messages -c user_id -f 'feat$1gram$messages$user_id$16to16' 'feat$2gram$messages$user_id$16to16' --combine_feat_tables 1to2gram

	# Extracts 1grams then 2grams, then combines the two new feature tables into a new one.
	# Creates:
	#	feat$1gram$messages$user_id$16to16
	# 	feat$2gram$messages$user_id$16to16
	# 	feat$1to2gram$messages$user_id$16to16
	dlatkInterface.py -d 2004blogs -t messages -c user_id --add_ngrams -n 1 2 --combine_feat_tables 1to2gram
