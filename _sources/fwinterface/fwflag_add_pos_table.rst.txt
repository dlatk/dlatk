.. _fwflag_add_pos_table:
===============
--add_pos_table
===============
Switch
======

--add_pos_table

Description
===========

Extract POS usage from the message table

Argument and Default Value
==========================

None

Details
=======

This switch uses the result from add_parses or add_tweetpos, and extracts POS usage from the POS tagged table.
Splits the POS tagged message on whitespace, then just sums up the occurrences of different POS.
Similar to add_pos_ngram_table but doesn't save the ngram itself.

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


	# Simply extracts POS from the message table "msgs_r10k"
	# Uses the fact that msgs_r10k_pos exists.
	# Creates: feat$pos$msgs_r10k$user_id$16to16
	dlatkInterface.py -d fb20 -t msgs_r10k -c user_id --add_pos_table` 