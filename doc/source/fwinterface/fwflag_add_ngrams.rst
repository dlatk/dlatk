.. _fwflag_add_ngrams:
============
--add_ngrams
============
Switch
======

--add_ngrams [-n N [N2...] ]

Description
===========

Extract ngrams (words and phrases) from a message table

Argument and Default Value
==========================

by default (i.e. if -n isn’t present), N is set to 1.

Details
=======

This switch tokenizes the given message table into N-grams and then aggregates them over each group_id. A n-gram is either a single word or token (n=1; e.g. “hello”, “:)”, “youtube.com”) or a phrase (n>1; e.g. “happy birthday”, “changes saved”). Once all of a group’s messages are split into n-grams , the n-grams  are counted (i.e. summed) (both raw count and relative frequency over that group_id) and inserted into a feature table that will be named something like “feat$ngram$message_table$group_id$16to16” (where the italicized items are determined by your command setup)
Note that the n-grams  are all lowercased.

By default spaces are shrunk and newlines ('\n') are replaced by '<newline>'.

The feature table will have the following columns (at least):

* group_id whatever values that the message table has in the :doc:`fwflag_c` column (is an index)
* feat ngrams (utf8mb4 encoded usually; also an index)
* value raw count of how many times the ngram appears in the group
* group_norm normalized frequency of ngram in group

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

Optional Switches:

* :doc:`fwflag_anscombe`, :doc:`fwflag_sqrt`, :doc:`fwflag_log`, :doc:`fwflag_boolean` 
* :doc:`fwflag_combine_feat_tables` 
* :doc:`fwflag_feat_occ_filter` 
* :doc:`fwflag_feat_colloc_filter` 
* :doc:`fwflag_no_metafeats` 

Example Commands
================

.. code-block:: bash

	# Simply extracts 1grams from the message table “messages”
	# Creates: feat$1gram$messages$user_id$16to16
	dlatkInterface.py -d 2004blogs -t messages -c user_id --add_ngrams

	# Extracts 1grams then 2grams, then combines the two new feature tables into a new one.
	# Creates:
	#	feat$1gram$messages$user_id$16to16
	# 	feat$2gram$messages$user_id$16to16
	# 	feat$1to2gram$messages$user_id$16to16
	dlatkInterface.py -d 2004blogs -t messages -c user_id --add_ngrams -n 1 2 --combine_feat_tables 1to2gram

To extract only the 1-grams that appear in the lexicon table ANEW (not really a collocation table; off-label use of :doc:`fwflag_use_collocs`): 

.. code-block:: bash

	dlatkInterface.py -d fbtrust -t messagesEn -c user_id --add_ngrams --use_collocs --colloc_table ANEW --colloc_column term --feature_type_name ANEWterms

