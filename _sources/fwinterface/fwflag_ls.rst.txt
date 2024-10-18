.. _fwflag_ls:
=====================
--show_feature_tables
=====================
Switch
======

--show_feature_tables

Description
===========

See all available feature tables in given database, message table and correl field.

SQL style wildcards ('%') will work with both :doc:`fwflag_t` and :doc:`fwflag_c` to see feature tables across message table or groupings.

Aliases: --show_feat_tables and --ls

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t`, :doc:`fwflag_c`

Example Commands
================

See user level features for the message table *msgs*:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --show_feature_tables

	SQL QUERY: SHOW TABLES FROM dla_tutorial LIKE 'feat$%$msgs$user_id$%' 
	Found 13 available feature tables
	feat$1gram$msgs$user_id$16to1
	feat$1gram$msgs$user_id$16to16
	feat$1gram$msgs$user_id$16to16$0_01
	feat$1to3gram$msgs$user_id$16to16
	feat$1to3gram$msgs$user_id$16to16$0_05
	feat$1to3gram$msgs$user_id$16to16$pmi6_0
	feat$2gram$msgs$user_id$16to16
	feat$3gram$msgs$user_id$16to16
	feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16
	feat$meta_1gram$msgs$user_id$16to1
	feat$meta_1gram$msgs$user_id$16to16
	feat$meta_2gram$msgs$user_id$16to16
	feat$meta_3gram$msgs$user_id$16to16

See features extracted at any level for the message table *msgs*:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c '%' --show_feature_tables

	SQL QUERY: SHOW TABLES FROM dla_tutorial LIKE 'feat$%$msgs$user_id$%' 
	Found 13 available feature tables
	feat$1gram$msgs$message_id$16to1
	feat$1gram$msgs$user_id$16to1
	feat$1gram$msgs$user_id$16to16
	feat$1gram$msgs$user_id$16to16$0_01
	feat$1to3gram$msgs$user_id$16to16
	feat$1to3gram$msgs$user_id$16to16$0_05
	feat$1to3gram$msgs$user_id$16to16$pmi6_0
	feat$2gram$msgs$user_id$16to16
	feat$3gram$msgs$user_id$16to16
	feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16
	feat$meta_1gram$msgs$message_id$16to1
	feat$meta_1gram$msgs$user_id$16to1
	feat$meta_1gram$msgs$user_id$16to16
	feat$meta_2gram$msgs$user_id$16to16
	feat$meta_3gram$msgs$user_id$16to16
