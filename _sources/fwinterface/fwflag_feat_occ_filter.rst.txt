.. _fwflag_feat_occ_filter:
=================
--feat_occ_filter
=================
Switch
======

--feat_occ_filter

Description
===========

Filter features based on how commonly they are used

Argument and Default Value
==========================

There is no default value.

Details
=======

Group ratio is set by :doc:`fwflag_set_p_occ` RATIO_OF_GROUPS. Then feat_occ_filter filters features so as to keep only those features which are used by RATIO_OF_GROUPS number of groups or more. The missing features are aggregated into a feature called <OOV> which contains the value and group norm data for all the missing features.  


Other Switches
==============

Required Switches:

* :doc:`fwflag_f`
* :doc:`fwflag_set_p_occ` RATIO_OF_GROUPS

Example Commands
================

.. code-block:: bash


	# Extract ngrams and filter in one command
	dlatkInterface.py -d fb22 -t msgsEn_r5k -c user_id --add_ngrams -n 1 2 3 --combine_feat_tables 1to3gram --feat_occ_filter --set_p_occ 0.05


	# Add a filter to a table that was generated without using collocs
	dlatkInterface.py -d fb22 -t msgsEn_r5k -c user_id -f 'feat$1to3gram$msgsEn_r5k$user_id$16to16' --feat_occ_filter --set_p_occ 0.05


	# Add a filter to a table that was generated using collocs, override the default group_freq_thresh value
	dlatkInterface.py -d fb22 -t msgsEn_r5k -c user_id -f 'feat$colloc$msgsEn_r5k$user_id$16to16' --word_table 'feat$colloc$msgsEn_r5k$user_id$16to16 --feat_occ_filter --set_p_occ 0.05 --group_freq_thresh 50