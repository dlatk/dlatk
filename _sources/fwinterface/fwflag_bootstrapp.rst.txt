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

* :doc:`fwflag_correlate`

Optional Switches: 

* :doc:`fwflag_no_correction`
* :doc:`fwflag_p_correction`
* :doc:`fwflag_outcome_controls` 

Example Commands
================

.. code-block:: bash


	dlatkInterface.py -d tester7 -t statuses_er1 -c study_code --group_freq_thresh 500 \ 
	-f 'feat$cat_ser1_f2_200_cp_w$statuses_er1_655$study_code$16to16' --outcome_table outcomesFinal \ 
	--outcomes DM_UNCOMP --correlate --rmatrix \ 
	--controls sex_int isWhite isBlack isHispanic ageTercile0 ageTercile1 ageTercile2 \ 
	--AUC --bootstrapp 10000 --no_correction --csv --output_name OUTPUT