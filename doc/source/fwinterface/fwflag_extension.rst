.. _fwflag_extension:
=============
--extension
=============
Switch
======

--view_tables string

Description
===========

String added to the end of the feature table name. Typical use case is when creating tables from non-standard word tables.


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t`, :doc:`fwflag_c`
* :doc:`fwflag_add_ngrams` or :doc:`fwflag_add_lex_table`

Example Commands
================

Create the table *feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16$0_01* using a non-default word table *feat$1gram$msgs$16to16$0_01*:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_lex_table -l met_a30_2000_cp --weighted_lexicon --extension '0_01' --word_table 'feat$1gram$msgs$16to16$0_01'


