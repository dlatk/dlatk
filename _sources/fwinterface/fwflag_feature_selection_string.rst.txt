.. _fwflag_feature_selection_string:
==========================
--feature_selection_string
==========================
Switch
======

--feature_selection_string NAME (or --feat_selection_string NAME)

Description
===========

Select feature selection pipeline used in prediction

Argument and Default Value
==========================

NAME must be any of the featureSelectionString strings found in regressionPredictor.py. There is no default value.

Details
=======


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` 

Optional Switches:

* :doc:`fwflag_combo_test_regression`
* :doc:`fwflag_control_adjust_outcomes_regression`
* --control_adjust_reg
* --predict_cv_to_feats
* --predict_combo_to_feats
* --predict_regression_all_to_feats
* :doc:`fwflag_combo_test_classifiers` 

Example Commands
================

.. code-block:: bash

	# train a language model for age
	dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id -f 'feat$cat_met_a30_2000_cp_w$msgs_xxx$user_id$16to16' 'feat$1gram$msgs_xxx$user_id$16to16' --outcome_table masterstats_r500 --group_freq_thresh 500 --outcomes demog_age --train_regression --model ridgefirstpasscv --feat_selection_string 'SelectFwe(f_regression, alpha=60.0)'
