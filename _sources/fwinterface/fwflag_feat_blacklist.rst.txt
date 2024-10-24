.. _fwflag_feat_blacklist:
================
--feat_blacklist
================
Switch
======

--feat_blacklist

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

* :doc:`fwflag_blacklist`

Example Commands
================

.. code-block:: bash


	#Correlation with all features in feature table other than agr and con:
	dlatkInterface.py -d tester7 -t statuses_er1 -c study_code -f 'feat$p_ridg_lbp_oceangft500$statuses_er1$study_code' --group_freq_thresh 0 --outcome_table outcomesFinal --outcomes DEPRESSION DM_UNCOMP EATING_DIS GI_SXS PREGNANCY PSYCHOSES STI --outcome_controls sex_int isBlack isWhite isHispanic ageTercile0 ageTercile1 ageTercile2 --correlate --rmatrix --blacklist --feat_blacklist agr con --output_name ./output 
