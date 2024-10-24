.. _fwflag_add_pos_ngram_table:
=====================
--add_pos_ngram_table
=====================
Switch
======

--add_pos_ngram_table

Description
===========

Extract 1gram/POS from the message table

Argument and Default Value
==========================

None

Details
=======

This switch uses the result from add_parses or add_tweetpos, and extracts ngrams from the POS tagged table.
Splits the POS tagged message on whitespace, then lowercases the word but not the POS.
Similar to add_ngrams but appends the POS that the ngram was used as.

Note that this can't be used with :doc:`fwflag_n`, it only works with n = 1.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

Optional Switches:

* :doc:`fwflag_anscombe`, :doc:`fwflag_sqrt`, :doc:`fwflag_log`, :doc:`fwflag_boolean`
* :doc:`fwflag_feat_occ_filter`
* :doc:`fwflag_feat_colloc_filter` 

Example Commands
================

.. code-block:: bash


	# Simply extracts 1gram (with POS) from the message table "msgs_r10k"
	# Uses the fact that msgs_r10k_pos exists.
	# Creates: feat$1gram_pos$msgs_r10k$user_id$16to16
	dlatkInterface.py -d fb20 -t msgs_r10k -c user_id --add_pos_ngram_table