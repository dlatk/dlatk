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
:doc:`fwflag_blacklist` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


Correlation with all features in feature table other than agr and con:
fwInterface.py :doc:`fwflag_d` tester7 :doc:`fwflag_t` statuses_er1 :doc:`fwflag_c` study_code :doc:`fwflag_f` 'feat$p_ridg_lbp_oceangft500$statuses_er1$study_code' :doc:`fwflag_group_freq_thresh` 0 :doc:`fwflag_outcome_table` outcomesFinal :doc:`fwflag_outcomes` DEPRESSION DM_UNCOMP EATING_DIS GI_SXS PREGNANCY PSYCHOSES STI :doc:`fwflag_outcome_controls` sex_int isBlack isWhite isHispanic ageTercile0 ageTercile1 ageTercile2 :doc:`fwflag_correlate` :doc:`fwflag_rmatrix` :doc:`fwflag_blacklist` :doc:`fwflag_feat_blacklist` agr con :doc:`fwflag_output_name` ./output 
