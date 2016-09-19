.. _fwflag_classification_to_lexicon:
===========================
--classification_to_lexicon
===========================
Switch
======

--classification_to_lexicon

Description
===========

Extracts the coefficients from a classification model and turns them into a lexicon.

Argument and Default Value
==========================

Name of the lexicon to be created.

Details
=======

Use this switch to create a lexicon from a classification model. Either create the lexicon from a previously created model (using :doc:`fwflag_load_model`) or create a model using :doc:`fwflag_train_classifiers`. The name of the lexicon will be dd_ARGUMENT ; dd indicates the lexicon was data driven.

IMPORTANT: When creating the model, use :doc:`fwflag_no_standardize` or the model will not make any sense.

Only use classification algorithms that have linear coefficients (i.e by choosing the right :doc:`fwflag_model`), because the lexicon extraction equation won't make sense otherwise. This functionality hasn't totally been validated with advanced feature selection, so beware.
Also note that the coefficients won't be efficient to distinguish what features best characterize the outcomes looked at, use :doc:`fwflag_correlate` or other univariate techniques to get at that type of insight.

NOTE: This only works with binary classification, so if you want to do this with multiple classes (>2), create variables that are binary.


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` :doc:`fwflag_no_standardize` Needs one of these two switches:
:doc:`fwflag_train_classifiers` :doc:`fwflag_load_model` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Trains a classification model to predict gender for users from 1grams, without standardizing
 # Will save the model to a picklefile called deleteMeGender.pickle, and create a lexicon called dd_testGenderLex
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` gender :doc:`fwflag_train_classifiers` :doc:`fwflag_save_model` :doc:`fwflag_picklefile` deleteMeGender.pickle 
 :doc:`fwflag_no_standardize` :doc:`fwflag_classification_to_lexicon` testGenderLex

 # Given a model that was previously made, this turns the model into a lexicon called dd_testAgeLex
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_load_model` :doc:`fwflag_picklefile` deleteMeGender.pickle 
 :doc:`fwflag_classification_to_lexicon` testGenderLex
References

Sap et al. (2014) - Developing Age and Gender Predictive Lexica over Social Media