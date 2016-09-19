.. _fwflag_all_controls_only:
===================
--all_controls_only
===================
Switch
======

--all_controls_only

Description
===========

Works similarly to --control_combo_sizes, but tells the classification or regression predictor to only train using all controls, so giving this switch is equivalent to

Argument and Default Value
==========================

where N is the number of controls specified with --outcome_controls.

Details
=======


Other Switches
==============

Required Switches:
:doc:`fwflag_outcome_controls` One of these:
:doc:`fwflag_combo_test_classifiers` OR
:doc:`fwflag_combo_test_regression` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


Only runs including all controls (and no controls, which is done by default):
fwInterface.py :doc:`fwflag_d` tester7 :doc:`fwflag_t` statuses_er1 :doc:`fwflag_c` study_code :doc:`fwflag_group_freq_thresh` 500 :doc:`fwflag_f` 'feat$cat_statuses_er1_cp_w$statuses_er1$study_code$16to16' :doc:`fwflag_outcome_table` outcomesFinal_femOnly :doc:`fwflag_outcomes` PREGNANCY :doc:`fwflag_controls` isWhite isBlack isHispanic ageTercile0 ageTercile1 ageTercile2 age :doc:`fwflag_combo_test_classif` :doc:`fwflag_all_controls_only` :doc:`fwflag_model` lr :doc:`fwflag_folds` 10 :doc:`fwflag_csv` :doc:`fwflag_output_name` ./output
