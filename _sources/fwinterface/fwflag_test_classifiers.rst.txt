.. _fwflag_test_classifiers:
==================
--test_classifiers
==================
Switch
======

--test_classifiers

Description
===========

Splits the data into test and training set, trains a classifier on training set and predicts it on the test set.

Argument and Default Value
==========================

None

Details
=======

This switch split the data into test (1/5) and training (4/5), then creates a classification model (aka a classifier) on the training data only. It then predicts the outcome class for the data in the test set, and yields accuracies for the created model. This technique is called out of sample prediction, and is used to avoid over:doc:`fwflag_fitting`. 
It is usually better to either use :doc:`fwflag_combo_test_classifiers`, which does the same thing as :doc:`fwflag_test_classifiers` but multiple times. Alternatively, you can manually create a test/training set by splitting your data in MySQL. If you're doing this, it's preferable to put "wordy" users in the training set, to boost the accuracy.


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` Optional Switches:
:doc:`fwflag_group_freq_thresh` :doc:`fwflag_no_standardize` :doc:`fwflag_model` :doc:`fwflag_sparse` etc.

Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Trains a classifier on 4/5ths of the users to predict their gender from 1grams
 # then predicts the genders of the remaining 1/5th of users, and compares the predicted genders to the real ones.
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` gender :doc:`fwflag_test_classifiers` 