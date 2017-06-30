.. _fwflag_add_ngrams_from_tokenized:
===========================
--add_ngrams_from_tokenized
===========================
Switch
======

--add_ngrams_from_tokenized [-n N [N2...] ]

Description
===========

Extract ngrams (words and phrases) from an already-tokenized message table

Argument and Default Value
==========================

by default (i.e. if -n isn’t present), N is set to 1.

Details
=======

This switch counts the tokens (single token if n=1, sets of adjacent tokens if n>1) in the given tokenized message table then aggregates them over each group_id. A ngram is either a single word or token (n=1; e.g. “hello”, “:)”, “youtube.com”) or a phrase (n>1; e.g. “happy birthday”, “changes saved”). The ngrams are counted (i.e. summed) (both raw count and relative frequency over that group_id) and inserted into a feature table that will be named something like “feat$ngram$message_table$group_id$16to16”. Note that the ngrams are all lowercased.

The feature table will have the following columns (at least):
group_id whatever values that the message table has in the :doc:`fwflag_c` column (is an index)
feat ngrams (utf8mb4 encoded usually; also an index)
value raw count of how many times the ngram appears in the group
group_norm normalized frequency of ngram in group





Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

Optional Switches:

* :doc:`fwflag_anscombe`, :doc:`fwflag_sqrt`, :doc:`fwflag_log`, :doc:`fwflag_boolean` 
* :doc:`fwflag_combine_feat_tables` 
* :doc:`fwflag_feat_occ_filter` 
* :doc:`fwflag_feat_colloc_filter` 

Example Commands
================

.. code-block:: python


	# Simply extracts 1grams from the message table "messages"
	# Creates: feat$1gram$messages$user_id$16to16
	./dlatkInterface.py -d 2004blogs -t messages_tok -c user_id --add_ngrams_from_tokenized
	
	# Extracts 1grams then 2grams, then combines the two new feature tables into a new one.
	# Creates:
	#	feat$1gram$messages$user_id$16to16
	# 	feat$2gram$messages$user_id$16to16
	# 	feat$1to2gram$messages$user_id$16to16
	dlatkInterface.py -d 2004blogs -t messages_tok -c user_id --add_ngrams_from_tokenized -n 1 2 --combine_feat_tables 1to2gram
