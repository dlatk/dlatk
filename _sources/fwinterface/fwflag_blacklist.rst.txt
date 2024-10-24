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

* :doc:`fwflag_feat_blacklist` 

Optional Switches:

* :doc:`fwflag_loessplot`
* :doc:`fwflag_correlate`
* :doc:`fwflag_combo_rmatrix`
* :doc:`fwflag_multir`
* :doc:`fwflag_interaction_ddla` 

Example Commands
================

.. code-block:: bash


	dlatkInterface.py -d tester7 -t statuses_er1 -c study_code -f 'feat$p_ridg_lbp_oceangft500$statuses_er1$study_code' --group_freq_thresh 0 --outcome_table outcomesFinal --outcomes DEPRESSION DM_UNCOMP EATING_DIS GI_SXS PREGNANCY PSYCHOSES STI --outcome_controls sex_int isBlack isWhite isHispanic ageTercile0 ageTercile1 ageTercile2 --correlate --rmatrix --blacklist --feat_blacklist agr con --output_name ./output 
