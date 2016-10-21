.. _fwflag_bootstrapp:
============
--bootstrapp
============
Switch
======

--bootstrapp

Description
===========

Currently (2015-07-24) only implemented for --AUC switch of --correlate

Argument and Default Value
==========================

Non-parametric significance test. On each iteration, shuffles predictor variable relative to outcome (and, if given, outcome controls) to get null distribution of AUC values. p-value is reported as percentile of actual value relative to the null distribution (i.e., percentage of values higher than real value)

Details
=======


Other Switches
==============

Required Switches:
:doc:`fwflag_correlate` Optional Switches: 
:doc:`fwflag_no_bonferroni` :doc:`fwflag_p_correction` :doc:`fwflag_outcome_controls` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 fwInterface.py :doc:`fwflag_d` tester7 :doc:`fwflag_t` statuses_er1 :doc:`fwflag_c` study_code :doc:`fwflag_group_freq_thresh` 500 \ 
 :doc:`fwflag_f` 'feat$cat_ser1_f2_200_cp_w$statuses_er1_655$study_code$16to16' :doc:`fwflag_outcome_table` outcomesFinal \ 
 :doc:`fwflag_outcomes` DM_UNCOMP :doc:`fwflag_correlate` :doc:`fwflag_rmatrix` \ 
 :doc:`fwflag_controls` sex_int isWhite isBlack isHispanic ageTercile0 ageTercile1 ageTercile2 \ 
 :doc:`fwflag_AUC` :doc:`fwflag_bootstrapp` 10000 :doc:`fwflag_no_bonferroni` :doc:`fwflag_csv` :doc:`fwflag_output_name` OUTPUT