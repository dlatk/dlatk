.. _fwflag_interaction_ddla_pvalue:
=========================
--interaction_ddla_pvalue
=========================
Switch
======

--interaction_ddla_pvalue

Description
===========

Sets significance threshold for whitelisting features in --interaction_ddla

Argument and Default Value
==========================

A possible p-value. Default = 0.001.

Details
=======

See :doc:`fwflag_interaction_ddla` for more explanations.

Other switches
==============

Required switches:
* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, 
* :doc:`fwflag_f`, 
* :doc:`fwflag_outcome_table`, 
* :doc:`fwflag_outcomes`,
* :doc:`fwflag_interaction_ddla` 

Example Commands
================
.. code-block:: doc:`fwflag_block`:: python


 	# Finds the features that have a significant (p<.05) interaction between Extreme_Red_10000 (binary voting behavior) and the LIWC categories.
 	# Using this list of features, it then does DLA (see :doc:`fwflag_correlate`) with controls twice:
 	# Once on the groups that are republican (i.e. where Extreme_Red_10000 = 1) and once on the democrats
 	~/fwInterface.py :doc:`fwflag_d` twitterGH :doc:`fwflag_t` messages_en :doc:`fwflag_c` cty_id :doc:`fwflag_group_freq_thresh` 20000 :doc:`fwflag_f` 'feat$cat_LIWC2007$messages_en$cty_id$16to16' 
 	:doc:`fwflag_outcome_table` countyVotingSM2 :doc:`fwflag_outcomes` overall_LS :doc:`fwflag_interaction_ddla` Extreme_Red_10000 :doc:`fwflag_outcome_controls` Median_Age 
 percent_white log_mean_income percent_bachelors :doc:`fwflag_output_name` ER10k.LIWC.interaction :doc:`fwflag_rmatrix` :doc:`fwflag_sort` :doc:`fwflag_no_bonf` :doc:`fwflag_csv` :doc:`fwflag_interaction_ddla_p` 0.05
