.. _fwflag_mediators:
===========
--mediators
===========
Switch
======

--mediators

Description
===========

The mediator variable in mediation analysis

Argument and Default Value
==========================

Pass a space separated list of names of variables found in the feature or outcome table. By default mediators are located in the feature table. If --feat_as_path_start, --feat_as_outcome, --feat_as_control  or --no_features is used then the mediators need to be in the outcome table.

Details
=======Contents [hide]Switch
Description
    Argument and Default Value
    
Other Switches
==============Switch

:doc:`fwflag_mediators` 
Description

The mediator variable in mediation analysis

Argument and Default Value

Pass a space separated list of names of variables found in the feature or outcome table. By default mediators are located in the feature table. If :doc:`fwflag_feat_as_path_start`, :doc:`fwflag_feat_as_outcome`, :doc:`fwflag_feat_as_control`  or :doc:`fwflag_no_features` is used then the mediators need to be in the outcome table.

Note: to consider each feature as a mediator you must do not use this switch. You also cannot use: :doc:`fwflag_feat_as_path_start`, :doc:`fwflag_feat_as_outcome`, :doc:`fwflag_feat_as_control`  or :doc:`fwflag_no_features` 


Other Switches
==============

Required Switches:
:doc:`fwflag_mediation` :doc:`fwflag_outcomes` OUTCOME_1 ... OUTCOME_K
:doc:`fwflag_outcome_table` OUTCOME_TABLE_NAME
Optional Switches:
:doc:`fwflag_feat_as_path_start` :doc:`fwflag_f` FEATURE_TABLE_NAME