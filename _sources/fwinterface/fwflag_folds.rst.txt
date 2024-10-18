.. _fwflag_folds:
=======
--folds
=======
Switch
======

--folds

Description
===========

Number of folds for functions that run n-fold cross-validation.

Argument and Default Value
==========================

Argument is integer representing number of folds. Default number of folds is 5.

Details
=======

The original sample is randomly partitioned into n subsamples, n-1 of which are used for training. This is repeated n times. The n trained models are then combined into a single model.


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes`

Optional Switches:

* :doc:`fwflag_combo_test_regression`
* --control_adjust_outcomes_regression
* --control_adjust_reg
* --predict_cv_to_feats
* --predict_combo_to_feats
* --predict_regression_all_to_feats
* :doc:`fwflag_combo_test_classifiers` 

Example Commands
================

.. code-block:: bash


	# Runs 10:doc:`fwflag_fold` cross validation on predicting the users' genders from 1grams.
	# This essentially will tell you how well your model & features do at predicting gender.
	# Splits the data in 10 chunks, for each chunk training a model on the remaining 9 chunks.
	dlatkInterface.py -d fb20 -t messages_en -c user_id -f 'feat$1gram$messages_en$user_id$16to16$0_01' --outcome_table masterstats_andy_r10k --outcomes gender --combo_test_classifiers --model linear-svc --folds 10
