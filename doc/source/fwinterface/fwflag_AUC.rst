.. _fwflag_AUC:
=====
--AUC
=====
Switch
======

--AUC, --auc

Description
===========

Use logistic regression model area under curve (AUC) for --correlate instead of linear regression/correlation [only works with binary outcome values]. "Curve" refers to receiver operating characteristic curve.

Argument and Default Value
==========================

Output value for each predictor feature will range from 0.5 (completely non-predictive) to 1.0 (perfectly predictive).

Details
=======


Other Switches
==============

Required Switches:

Optional Switches: :doc:`fwflag_bootstrapp` 

Example Commands
================
.. code:doc:`fwflag_block`:: python


 fwInterface.py :doc:`fwflag_d` tester7 :doc:`fwflag_t` statuses_er1 :doc:`fwflag_c` study_code :doc:`fwflag_group_freq_thresh` 500 \ 
 :doc:`fwflag_f` 'feat$cat_ser1_f2_200_cp_w$statuses_er1_655$study_code$16to16' \ 
 :doc:`fwflag_outcome_table` outcomesFinal :doc:`fwflag_outcomes` DM_UNCOMP \ 
 :doc:`fwflag_correlate` :doc:`fwflag_rmatrix` :doc:`fwflag_controls` sex_int isWhite isBlack isHispanic ageTercile0 ageTercile1 ageTercile2 \ 
 :doc:`fwflag_auc` :doc:`fwflag_no_bonferroni` :doc:`fwflag_csv` :doc:`fwflag_output_name` OUTPUT