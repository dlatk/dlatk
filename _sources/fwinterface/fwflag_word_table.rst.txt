.. _fwflag_word_table:
============
--word_table
============
Switch
======

--word_table

Description
===========

Table containing "word usage" per group

Argument and Default Value
==========================

Table name in MySQL. Usually starts with feat$....

Details
=======

This switch specifies what the infrastructure will use as the table that contains a group's word usage. By default (i.e. if not specified) this table is feat$1gram$MESSAGE_TABLE$GROUP_ID$16to16.

Use cases:

* :doc:`fwflag_add_lex_table` If your lexicon was build on 1to3grams, supply the 1to3gram table with this switch, it will then match all ngrams. If not used, the lexicon will only count things that are 1 grams in the lexicon.
* If you need to use :doc:`fwflag_group_freq_thresh` but you don't have a complete 1gram table by the feat$1gram$MESSAGE_TABLE$GROUP_ID$16to16 format (i.e., if that would be too large and you only have a feat_occ_filtered table), you can supply your word table here.

Example Commands
================

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_lex_table -l met_a30_2000 --weighted_lexicon  --word_table 'feat$1gram$msgs$user_id$16to16$0_01'