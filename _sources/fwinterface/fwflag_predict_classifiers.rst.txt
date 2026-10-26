.. _fwflag_predict_classifiers:
=====================
--predict_classifiers
=====================
Switch
======

--predict_classifiers

Description
===========

Predicts the outcomes and compares the predicted values to the actual ones.

Argument and Default Value
==========================

None

Details
=======

Given a model (:doc:`fwflag_load_model`), this switch will predict the outcome classes on the groups given in the outcome table and compare them to the actual classes in the outcome table.
Make sure the feature tables are in the same order as they were when the model was created.

The output will contains some of the following things:
confusion matrix See here.
precision and recall See here
and this line (numbers are examples):
 FOLD ACC: 0.8839 (mfclass_acc: 0.6186); mfclass: 1; auc: 0.8749
ACC percent classified correctly
mfclass_acc accuracy if we predicted the main class every time (baseline)
mfclass "Most frequent class" i.e. the class that has the most groups in it.
auc Area Under the Curve (the ROC curve). See here for more explanations.

Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` :doc:`fwflag_load_model` and :doc:`fwflag_picklefile` Optional Switches:
:doc:`fwflag_classification_to_lexicon` :doc:`fwflag_group_freq_thresh` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Loads the classification model in deleteMeGender.pickle, and uses the features to predict the gender 
 # of users in masterstats_andy_r10k, and compares the predicted genders to the actual ones in the table.
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` gender :doc:`fwflag_load_model` :doc:`fwflag_picklefile` deleteMeGender.pickle 
 :doc:`fwflag_predict_classifiers` 