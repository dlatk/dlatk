.. _fwflag_blacklist:
===========
--blacklist
===========
Switch
======

--blacklist

Description
===========

Switch to enable blacklisting of features in correlation-type analyses.

Argument and Default Value
==========================

None

Details
=======

Requires :doc:`fwflag_feat_blacklist`. Argument of that switch gives either feature table as a blacklist (distinct features of that table) or arguments are a list of features to blacklist.

See also whitelist and feat_whitelist


Other Switches
==============

Required Switches:
:doc:`fwflag_feat_blacklist` Optional Switches:
:doc:`fwflag_loessplot` :doc:`fwflag_correlate` :doc:`fwflag_combo_rmatrix`? :doc:`fwflag_multir`? :doc:`fwflag_interaction_ddla` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


fwInterface.py :doc:`fwflag_d` tester7 :doc:`fwflag_t` statuses_er1 :doc:`fwflag_c` study_code :doc:`fwflag_f` 'feat$p_ridg_lbp_oceangft500$statuses_er1$study_code' :doc:`fwflag_group_freq_thresh` 0 :doc:`fwflag_outcome_table` outcomesFinal :doc:`fwflag_outcomes` DEPRESSION DM_UNCOMP EATING_DIS GI_SXS PREGNANCY PSYCHOSES STI :doc:`fwflag_outcome_controls` sex_int isBlack isWhite isHispanic ageTercile0 ageTercile1 ageTercile2 :doc:`fwflag_correlate` :doc:`fwflag_rmatrix` :doc:`fwflag_blacklist` :doc:`fwflag_feat_blacklist` agr con :doc:`fwflag_output_name` ./output 
