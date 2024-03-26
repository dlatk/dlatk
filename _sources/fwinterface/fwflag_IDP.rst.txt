.. _fwflag_IDP:
=====
--IDP
=====
Switch
======

--IDP, --idp

Description
===========

Uses Informative Dirichlet Prior (see references) in correlations with binary outcomes instead of linear regression.

Argument and Default Value
==========================

None

Details
=======


Other Switches
==============

Required Switches:
:doc:`fwflag_correlate` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


fwInterface.py :doc:`fwflag_d` tester7 :doc:`fwflag_t` statuses_er1 :doc:`fwflag_c` study_code :doc:`fwflag_f` 'feat$1to3gram$statuses_er1$study_code$16to16$0_1' :doc:`fwflag_group_freq_thresh` 500 :doc:`fwflag_outcome_table` PDS_BMI_subj :doc:`fwflag_outcomes` BMI_avg_2bin BMI_range_2bin BMI_slope_2bin BMI_absslope_2bin BMI_avg_1q4q BMI_range_1q4q BMI_slope_1q4q BMI_absslope_1q4q :doc:`fwflag_outcome_controls` sex_int isBlack isWhite isHispanic ageTercile0 ageTercile1 ageTercile2 :doc:`fwflag_correlate` :doc:`fwflag_IDP` :doc:`fwflag_rmatrix` :doc:`fwflag_sort` :doc:`fwflag_tagcloud` :doc:`fwflag_make_wordclouds` :doc:`fwflag_output_name` 
