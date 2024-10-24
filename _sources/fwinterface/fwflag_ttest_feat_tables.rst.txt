.. _fwflag_ttest_feat_tables:
===================
--ttest_feat_tables
===================
Switch
======

--ttest_feat_tables

Description
===========

Paired t-test of group_norms for all features in two feature tables with group_id as observations

Argument and Default Value
==========================

None

Details
=======


Other Switches
==============

Required Switches:
Requires two feature tables to be given after :doc:`fwflag_f` Optional Switches:
:doc:`fwflag_group_freq_thresh` is applied to both feature tables; only group_ids present in both and passing GFT will be taken as valid observations
:doc:`fwflag_csv` writes output (t, p, and effect size d for each feature) to a file

Example Commands
================
.. code:doc:`fwflag_block`:: python


fwInterface.py :doc:`fwflag_d` tester7 :doc:`fwflag_t` statuses_er1 :doc:`fwflag_c` study_code :doc:`fwflag_f` 'feat$cat_statuses_er1_cp_w$statuses_6moBB$study_code$16to16' 'feat$cat_statuses_er1_cp_w$statuses_6moAB$study_code$16to16' :doc:`fwflag_ttest_feat_tables` :doc:`fwflag_group_freq_thresh` 100 :doc:`fwflag_csv` :doc:`fwflag_output_name` /localdata/patrick/pregDLA/ttest_res/before_after

