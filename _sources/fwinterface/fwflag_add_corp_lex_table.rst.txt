.. _fwflag_add_corp_lex_table:
====================
--add_corp_lex_table
====================
Switch
======

--add_corp_lex_table [-l LEX]

Description
===========

Add a lexicon-based feature table

Argument and Default Value
==========================

--add_corp_lex_table -l <LEX> (where <LEX> is the name of the table)

Details
=======

Given an existing 1gram table as your original feature table, this switch produces a new feature table where the features are now topics (as opposed to N-grams) based on the topics in the LEX table.

Unless :doc:`fwflag_f` is provided, the table feat$1gram$TABLE$GROUP_BY_FIELD$16to16 must be present.

New table is lex$feat_source$message_table$group$transform


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_l` :doc:`fwflag_weighted_lexicon` 

Optional Switches:

* :doc:`fwflag_f` [alternative n-gram table; can be used in place of word table]

Example Commands
================

.. code-block:: python

	# Creates the table: lex$cat_met_a30_2000_cp_w$primals_new$dp_id$16to16
	dlatkInterface.py -d primals -t primals_new -c dp_id --group_freq_thresh 40000 --add_corp_lex_table -l met_a30_2000_cp --weighted_lex