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


Example Commands
================
.. code:doc:`fwflag_block`:: python
 
 
 # For Classification 
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` gender :doc:`fwflag_combo_test_classifiers` :doc:`fwflag_folds` 10 
 :doc:`fwflag_stratify_folds` :doc:`fwflag_save_model` 
 :doc:`fwflag_picklefile` deleteMeGender.pickle

 # For Regression
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age :doc:`fwflag_combo_test_regression` :doc:`fwflag_folds` 10 
 :doc:`fwflag_stratify_folds` :doc:`fwflag_save_model` 
 :doc:`fwflag_picklefile` deleteMeAge.pickle