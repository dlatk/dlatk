.. _fwflag_stratify_folds:
==========
--stratify_folds
==========
Switch
======

--stratify_folds


Description
===========

This switch is used to specify that the data should be stratified into folds with respect to the outcome column. This will ensure that the outcome distribution of each fold is similar to the overall outcome distribution. 

Argument and Default Value
==========================

None

Details
=======

Can be used for both binary and continuous outcomes.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 
* :doc:`fwflag_combo_test_regression` or :doc:`fwflag_combo_test_classifiers`

Example Commands
================

.. code-block:: bash

    # Runs stratified 10-fold cross validation on predicting the users ages from 1grams.
    dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$16to16' --outcome_table blog_outcomes \
    --outcomes age --combo_test_regression --model ridgecv --folds 10 --stratify_folds