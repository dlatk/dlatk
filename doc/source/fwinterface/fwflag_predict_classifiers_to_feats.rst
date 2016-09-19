.. _fwflag_predict_classifiers_to_feats:
==============================
--predict_classifiers_to_feats
==============================
Switch
======

--predict_classifiers_to_feats

Description
===========

Predicts the outcome class and puts the predicted values into a SQL table

Argument and Default Value
==========================

Feature name for the SQL table.

Details
=======

Given a classification model (:doc:`fwflag_load_model`), this switch will predict the outcome class on the groups given in the outcome table and puts the values into a MySQL table. This is useful for a set of groups that you don't have the outcomes for, but you have a prediction model for it (here are the outcomes we can predict from language).
The table created will look like:
feat$p_modelType_ARGUMENT$message_table$group_id where modelType is the first 4 letter of the model name.
If you used linear:doc:`fwflag_svc` for instance, it will look like feat$p_line_ARGUMENT$message_table$group_id.

Make sure the features are in the right order (i.e. the order they were put into when creating the model). A good place to check for that is the name of the pickle file (if you're using a pre:doc:`fwflag_made` picklefile, like those in here)

For now, you need to make an output table that contains non null values for the outcomes & groups that you want predictions for, cause it uses the :doc:`fwflag_predict_classifiers` code to run this, which is why it also outputs comparisons between the values in the outcome table and the predicted outcomes.
This should be changed soon though, so stay tuned!


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` :doc:`fwflag_load_model` and :doc:`fwflag_picklefile` Optional Switches:
:doc:`fwflag_group_freq_thresh` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Loads the classifier in deleteMeGender.pickle, and uses the features to predict the genders of the users in 
 # masterstats_andy_r10k, and inserts those values into a table called feat$p_line_deleteMeGender$messages_en$user_id.
 # Note that it only predicts gender for those groups with non null gender values in masterstats_andy_r10k.
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` gender :doc:`fwflag_load_model` :doc:`fwflag_picklefile` deleteMeGender.pickle 
 :doc:`fwflag_predict_classifiers_to_feats` deleteMeGender
