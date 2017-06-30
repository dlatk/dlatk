.. _fwflag_correlate:
===========
--correlate
===========
Switch
======

--correlate

Description
===========

Correlates features with the given outcomes and print r-s to standard output.

Argument and Default Value
==========================

None

Details
=======

This is one of the flag that triggers the correlation code (just like :doc:`fwflag_rmatrix` or :doc:`fwflag_tagcloud`). If none of the :doc:`fwflag_outcome_controls` or :doc:`fwflag_outcome_interaction` flags are specified, a Pearson correlation will be done for every feature's group_norm and the outcome. See the specific pages for the types of analyses performed when there's controls or interaction variables.

Every p-value is by default bonferroni corrected, unless :doc:`fwflag_no_correction` is specified.

NOTE - group columns must match in type between the message table, feature table and outcome table!

The following pseudo-code is happening

.. code-block:: bash

	for feat in all_features:
		for outcome in outcomes:
	    	# x: column vector of group_norms for given feature
	    	# y: column vector of outcome values; aligned to x
	    	(r, p) = pearsonr(x,y)

:doc:`fwflag_correlate` prints out the following tuples to the stdout:

.. code-block:: bash

	("feature", (pearson-r, p-value, (confidence interval left, confidence interval right), number of groups/sample size, total count of "feature")

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` 

Optional Switches:

* :doc:`fwflag_group_freq_thresh`
* :doc:`fwflag_outcome_controls`
* :doc:`fwflag_outcome_interaction`
* :doc:`fwflag_rmatrix`
* :doc:`fwflag_no_correction`
* :doc:`fwflag_p_correction` [METHOD]
* :doc:`fwflag_AUC`
* :doc:`fwflag_spearman`
* :doc:`fwflag_IDP`
* :doc:`fwflag_bootstrapp` (as of 2015-07-24 only implemented with AUC)
* :doc:`fwflag_csv`
* :doc:`fwflag_sort`
* :doc:`fwflag_tagcloud`
* :doc:`fwflag_make_wordclouds`
* :doc:`fwflag_topic_tagcloud`
* :doc:`fwflag_whitelist`
* :doc:`fwflag_blacklist` 

Example Commands
================

.. code-block:: bash


	# Correlates LIWC lexical features with age and gender for every user in masterstats_andy_r10k 
	dlatkInterface.py -d fb20 -t messages_en -c user_id --outcome_table masterstats_andy_r10k --outcomes age gender -f 'feat$cat_LIWC2007$messages_en$user_id$16to16' --correlate 