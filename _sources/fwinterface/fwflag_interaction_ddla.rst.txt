.. _fwflag_interaction_ddla:
==================
--interaction_ddla
==================
Switch
======

--interaction_ddla

Description
===========

Finds features with significant interaction between language and argument, then does DLA on the two categories in the argument.

Argument and Default Value
==========================

A list of column names, separated by a space. There is no default value.

Details
=======

Building off of :doc:`fwflag_outcome_interaction`, finds features in :doc:`fwflag_f` that have a significant interaction with the argument (i.e. the features for which the beta for  is significant). It then performs DLA twice, one time per category in the argument, and for these runs it's equivalent to running DLA with outcomes and controls, with a feature whitelist. Except that in this case, only one command is needed, and the output will all be in one file.

Significance threshold is determined by the argument of :doc:`fwflag_interaction_ddla_pvalue`. 
The output will have two types of columns in the rmatrix:
INTER[...] values from the first step where the interaction was performed. Only the features with significant betas will be used in the next steps
[outcome]_0 or [outcome]_1 second and third steps, i.e. DLA with whitelist.

Other switches
==============

Required switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, 
* :doc:`fwflag_f`, 
* :doc:`fwflag_outcome_table`, 
* :doc:`fwflag_outcomes`
 
Example Commands
================
.. code-block:: python


 	# Finds the features that have a significant interaction between Extreme_Red_10000 (binary voting behavior) and the LIWC categories.
 	# Using this list of features, it then does DLA (see :doc:`fwflag_correlate`) with controls twice:
 	# Once on the groups that are republican (i.e. where Extreme_Red_10000 = 1) and once on the democrats
 	~/dlatkInterface.py :doc:`fwflag_d` twitterGH :doc:`fwflag_t` messages_en :doc:`fwflag_c` cty_id :doc:`fwflag_group_freq_thresh` 20000 :doc:`fwflag_f` 'feat$cat_LIWC2007$messages_en$cty_id$16to16' 
 	:doc:`fwflag_outcome_table` countyVotingSM2 :doc:`fwflag_outcomes` overall_LS :doc:`fwflag_interaction_ddla` Extreme_Red_10000 :doc:`fwflag_outcome_controls` Median_Age 
 percent_white log_mean_income percent_bachelors :doc:`fwflag_output_name` ER10k.LIWC.interaction :doc:`fwflag_rmatrix` :doc:`fwflag_sort` :doc:`fwflag_no_bonf` :doc:`fwflag_csv` :doc:`fwflag_interaction_ddla_p` 0.05
