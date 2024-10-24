.. _fwflag_combo_test_classifiers:
========================
--nfold_test_classifiers
========================
Switch
======

--nfold_test_classifiers or --comb_test_classifiers

Description
===========

Does K folds of splitting the data into test and training set, trains a model on training set and predicts it on the test set. K-fold cross validation.

Argument and Default Value
==========================

None

Details
=======

Similarly to :doc:`fwflag_test_classifiers`, this switch causes the data to be randomly spit in N chunks (where N is either 5 by default or defined by :doc:`fwflag_folds`). For each chunk, a classification model is trained on the remaining N-1 chunks and tested on this chunk (i.e. we see how well it performs).

After all chunks have been tested on, the accuracies and other metrics are averaged and printed out, which says something about the parameters and model chosen.

Note that all remarks about feature selection and model/parameter selection from :doc:`fwflag_train_classifiers` apply, so please read that section.

If you included :doc:`fwflag_controls` in your command, multiple K-fold CV will be done, one run for each of all the possible subsets of controls. Unless :doc:`fwflag_control_combo_sizes` or :doc:`fwflag_all_controls_only` is specified.

Output

Per fold, you will get a bunch of things printed to stdout. See :doc:`fwflag_predict_classifiers` for explanations.

At the end of the folds, you'll get something looking like this:

.. code-block:: bash

	{'gender': {(): {1: {'acc': 0.86367966775116722,
	                  'auc': 0.85393467507425691,
	                  ...

If there were controls included, you get 


.. code-block:: bash

	{'gender': {(): {1: {'acc': 0.898,
	                  ...}},
	         ('age',): {0: {'acc': 0.898,
	                        ...},
	                    1: {'acc': 0.898,
	                        ...}}
	}}

The first set of metrics ((): {1...) is the prediction performance of the language features alone, without any of the controls.

('age',) means age was included as a control in the prediction of age, and the first item in the dictionary ({0: {...}) is the performance using just the control values, no language, and then the ({1: {...}) is the performance with both controls and language. As you add controls, there will be 2n result dictionaries.


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes`
* :doc:`fwflag_pred_csv`

Optional Switches:

* :doc:`fwflag_model`
* :doc:`fwflag_no_standardize`
* :doc:`fwflag_folds`
* :doc:`fwflag_sparse`
* :doc:`fwflag_group_freq_thresh`
* :doc:`fwflag_all_controls_only`
* :doc:`fwflag_control_combo_sizes`
* :doc:`fwflag_no_lang` 

Example Commands
================

.. code-block:: bash


	# Runs 10-fold cross validation on predicting the users' genders from 1grams.
	# This essentially will tell you how well your model & features do at predicting gender.
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$messages_en$user_id$16to16$0_01' \
	--outcome_table blog_outcomes --outcomes gender --combo_test_classifiers --model linear-svc --folds 10
