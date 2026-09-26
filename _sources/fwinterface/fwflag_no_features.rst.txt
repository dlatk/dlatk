.. _fwflag_no_features:
=============
--no_features
=============
Switch
======

--no_features

Description
===========

To be used in Mediation analysis when all of the path start, mediator and outcome variables are in the outcome table. Whereas other mediation switches allow you to consider all features in a feature table (by leaving one of --path_starts, --mediators or --outcomes blank), there is no option for that with this switch.

Argument and Default Value
==========================

Required Switches:

Details
=======

To be used in Mediation analysis when all of the path start, mediator and outcome variables are in the outcome table. Whereas other mediation switches allow you to consider all features in a feature table (by leaving one of :doc:`fwflag_path_starts`, :doc:`fwflag_mediators` or :doc:`fwflag_outcomes` blank), there is no option for that with this switch.


Other Switches
==============

Required Switches:
:doc:`fwflag_mediation` :doc:`fwflag_path_starts` PATH_START_1 ... PATH_START_I
:doc:`fwflag_mediators` MEDIATOR_1 ... MEDIATOR_J
:doc:`fwflag_outcomes` OUTCOME_1 ... OUTCOME_K
Note: there are many other required and optional switches when running :doc:`fwflag_mediation`. 

Example Commands
================
.. code:doc:`fwflag_block`:: python


 # GENERAL COMMAND
 ./fwInterface.py :doc:`fwflag_d` DATABASE :doc:`fwflag_t` TABLE_NAME :doc:`fwflag_c` GROUP :doc:`fwflag_outcome_table` OUTCOME_TABLE_NAME :doc:`fwflag_mediation` \ 
 :doc:`fwflag_outcomes` OUTCOME_1 ... OUTCOME_N :doc:`fwflag_path_start` PATHSTART_1 ... PATHSTART_M  :doc:`fwflag_mediator` MEDIATOR_1 ... MEDIATOR_L :doc:`fwflag_no_features` 