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

.. code-block:: bash

	--control_combo_sizes N

where N is the number of controls specified with --outcome_controls.

Argument and Default Value
==========================

None

Details
=======


Other Switches
==============

Required Switches:

* :doc:`fwflag_outcome_controls` 
* One of these: :doc:`fwflag_combo_test_classifiers` OR :doc:`fwflag_combo_test_regression` 

Optional Switches:

* None

Example Commands
================

.. code-block:: bash


	# Only runs including all controls (and no controls, which is done by default):
	dlatkInterface.py -d tester7 -t statuses_er1 -c study_code --group_freq_thresh 500 -f 'feat$cat_statuses_er1_cp_w$statuses_er1$study_code$16to16' --outcome_table outcomesFinal_femOnly --outcomes PREGNANCY --controls isWhite isBlack isHispanic ageTercile0 ageTercile1 ageTercile2 age --combo_test_classif --all_controls_only --model lr --folds 10 --csv --output_name ./output
