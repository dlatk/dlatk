.. _fwflag_path_starts:
=============
--path_starts
=============
Switch
======

--path_starts

Description
===========

The independent or treatment variable in mediation analysis

Argument and Default Value
==========================

Pass a space separated list of names of variables found in the feature or outcome table. By default path starts are located in the outcome table. Use --feat_as_path_start to use variables from the feature table.

Details
=======Contents [hide]Switch
Description
    Argument and Default Value
    
Other Switches
==============Switch

:doc:`fwflag_path_starts` 
Description

The independent or treatment variable in mediation analysis

Argument and Default Value

Pass a space separated list of names of variables found in the feature or outcome table. By default path starts are located in the outcome table. Use :doc:`fwflag_feat_as_path_start` to use variables from the feature table. 

Note: if you want to use all features in a feature table as path starts you must use :doc:`fwflag_feat_as_path_start` and do NOT use :doc:`fwflag_path_starts`. No list of variables is needed.


Other Switches
==============

Required Switches:
:doc:`fwflag_mediation` :doc:`fwflag_outcomes` OUTCOME_1 ... OUTCOME_K
:doc:`fwflag_outcome_table` OUTCOME_TABLE_NAME
Optional Switches:
:doc:`fwflag_feat_as_path_start` :doc:`fwflag_f` FEATURE_TABLE_NAME