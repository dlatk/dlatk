.. _fwflag_keep_low_variance_outcomes:
============
--keep_low_variance_outcomes
============
Switch
======

``--keep_low_variance_outcomes`` or ``--keep_low_variance``

Description
===========

Keep any outcomes, controls or interactions that have low variance. 

Argument and Default Value
==========================

By default DLATK will calculate the variance for all outcomes, controls and interaction variables and remove if less than the default threshold. Use this flag to turn this feature off.

The default threshold is 0 and is set via a variable in ``dlaConstants.py``:

.. code-block:: python

	DEF_LOW_VARIANCE_THRESHOLD = 0.0

or can be changed via ``OutcomeGetter`` and ``OutcomeAnalyzer`` instance variables:

.. code-block:: python

	OutcomeGetter(..., low_variance_thresh=foo, ...)
	OutcomeAnalyzer(..., low_variance_thresh=foo, ...)

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

Optional Switches:

* Some correlation command: :doc:`fwflag_correlate`, :doc:`fwflag_logistic_reg`, etc.
* Some regression command: :doc:`fwflag_combo_test_regression`, :doc:`fwflag_predict_regression`, :doc:`fwflag_test_regression`, etc. 
* Some classification command: :doc:`fwflag_combo_test_classifiers`, :doc:`fwflag_predict_classifiers`, :doc:`fwflag_test_classifiers`, etc.

Example Commands
================

These are two toy examples where we correlate language features with gender but only consider males. You probably don't want to do this in practice.

.. code-block:: bash

	# run DLA over only males
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --outcome_table blog_outcomes \
	--outcomes gender -f 'feat$1gram$msgs$user_id$16to16' --correlate --where "gender = 0" --keep_low_variance

.. code-block:: bash

	# use 1grams to predict the gender of only males via 10-fold cross validation
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --outcome_table blog_outcomes \
	--outcomes gender -f 'feat$1gram$msgs$user_id$16to16' --combo_test_regression \
	--folds 10 --where "gender = 0" --keep_low_variance

