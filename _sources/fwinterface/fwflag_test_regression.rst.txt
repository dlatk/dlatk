.. _fwflag_test_regression:
=================
--test_regression
=================
Switch
======

--test_regression

Description
===========

Splits the data into test and training set, trains a model on training set and predicts it on the test set.

Argument and Default Value
==========================

None

Details
=======

This switch split the data into test (1/5) and training (4/5), then creates a regression model on the training data only. It then predicts the outcomes for the data in the test set, and yields accuracies for the created model. This technique is called out of sample prediction, and is used to avoid over:doc:`fwflag_fitting`. 
It is usually better to either use :doc:`fwflag_combo_test_regression`, which does the same thing as :doc:`fwflag_test_regression` but multiple times. Alternatively, you can manually create a test/training set by splitting your data in MySQL. If you're doing this, it's preferable to put "wordy" users in the training set, to boost the accuracy.


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` Optional Switches:
:doc:`fwflag_group_freq_thresh` :doc:`fwflag_no_standardize` :doc:`fwflag_model` :doc:`fwflag_sparse` etc.

Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Trains a regression model on 4/5ths of the users to predict their age from 1grams
 # then predicts the ages of the remaining 1/5th of users, and compares the predicted ages to the real ones.
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age :doc:`fwflag_test_regression` 