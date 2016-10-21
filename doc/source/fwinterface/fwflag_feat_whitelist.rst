.. _fwflag_feat_whitelist:
================
--feat_whitelist
================
Switch
======

--feat_whitelist

Description
===========

[feat1 feat2 ...] OR [feature table]

Argument and Default Value
==========================

if feature table is given, then read distinct features

Details
=======


Other Switches
==============

Required Switches:
:doc:`fwflag_print_tokenized_lines` OR
:doc:`fwflag_whitelist` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


Printing tokenized lines with 1 grams that were only used by 0.5% or more of study_codes:
fwInterface.py :doc:`fwflag_d` tester7 :doc:`fwflag_t` statuses_er1_655 :doc:`fwflag_print_tokenized_lines` ser1_filt.txt :doc:`fwflag_feat_whitelist` 'feat$1gram$statuses_er1_655$study_code$16to16$0_005'
With whitelist for only correlating subset of features (agr and con):
fwInterface.py :doc:`fwflag_d` tester7 :doc:`fwflag_t` statuses_er1 :doc:`fwflag_c` study_code :doc:`fwflag_f` 'feat$p_ridg_lbp_oceangft500$statuses_er1$study_code' :doc:`fwflag_group_freq_thresh` 0 :doc:`fwflag_outcome_table` outcomesFinal :doc:`fwflag_outcomes` DEPRESSION DM_UNCOMP EATING_DIS GI_SXS PREGNANCY PSYCHOSES STI :doc:`fwflag_outcome_controls` sex_int isBlack isWhite isHispanic ageTercile0 ageTercile1 ageTercile2 :doc:`fwflag_correlate` :doc:`fwflag_rmatrix` :doc:`fwflag_whitelist` :doc:`fwflag_feat_whitelist` agr con :doc:`fwflag_output_name` ./output 
