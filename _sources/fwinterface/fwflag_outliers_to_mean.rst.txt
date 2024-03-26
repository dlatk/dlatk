.. _fwflag_outliers_to_mean:
============
--outliers_to_mean
============
Switch
======

``--outliers_to_mean [OUTLIER_THRESHOLD]``

Description
===========

Set an outlier threshold. After standardization if absolute feature value is greater than threshold then set feature to mean value. 

Argument and Default Value
==========================

Default threshold is 2.5

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

Optional Switches:

* Some regression command: :doc:`fwflag_combo_test_regression`, :doc:`fwflag_predict_regression`, :doc:`fwflag_test_regression`, etc. 
* Some classification command: :doc:`fwflag_combo_test_classifiers`, :doc:`fwflag_predict_classifiers`, :doc:`fwflag_test_classifiers`, etc.

Example Commands
================

.. code-block:: bash

	# Runs 10-fold cross validation on predicting the users ages from 1grams.
	# Set outliers to the default value of 2.5
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$16to16' --outcome_table blog_outcomes \
	--outcomes age --combo_test_regression --model ridgecv --folds 10 --outliers_to_mean

.. code-block:: bash

	# Set the threshold to 3.5
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$16to16' --outcome_table blog_outcomes \
	--outcomes age --combo_test_regression --model ridgecv --folds 10 --outliers_to_mean 3.5

