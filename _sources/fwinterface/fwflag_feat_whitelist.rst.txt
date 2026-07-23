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

* :doc:`fwflag_print_tokenized_lines` OR :doc:`fwflag_whitelist` 


Example Commands
================

.. code-block:: bash


	#Printing tokenized lines with 1 grams that were only used by 0.5% or more of study_codes:
	dlatkInterface.py -d tester7 -t statuses_er1_655 --print_tokenized_lines ser1_filt.txt --feat_whitelist 'feat$1gram$statuses_er1_655$study_code$16to16$0_005'
	
	#With whitelist for only correlating subset of features (agr and con):
	dlatkInterface.py -d tester7 -t statuses_er1 -c study_code -f 'feat$p_ridg_lbp_oceangft500$statuses_er1$study_code' --group_freq_thresh 0 --outcome_table outcomesFinal --outcomes DEPRESSION DM_UNCOMP EATING_DIS GI_SXS PREGNANCY PSYCHOSES STI --outcome_controls sex_int isBlack isWhite isHispanic ageTercile0 ageTercile1 ageTercile2 --correlate --rmatrix --whitelist --feat_whitelist agr con --output_name ./output 
