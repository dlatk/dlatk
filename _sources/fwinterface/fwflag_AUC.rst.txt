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

Optional Switches: 

* :doc:`fwflag_bootstrapp` 

Example Commands
================

.. code-block:: bash


	daltkInterface.py -d tester7 -t statuses_er1 -c study_code --group_freq_thresh 500 \ 
	-f 'feat$cat_ser1_f2_200_cp_w$statuses_er1_655$study_code$16to16' \ 
	--outcome_table outcomesFinal --outcomes DM_UNCOMP \ 
	--correlate --rmatrix --controls sex_int isWhite isBlack isHispanic ageTercile0 ageTercile1 ageTercile2 \ 
	--auc --no_correction --csv --output_name OUTPUT