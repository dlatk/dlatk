.. _fwflag_regression_to_lexicon:
=======================
--regression_to_lexicon
=======================
Switch
======

--regression_to_lexicon

Description
===========

Extracts the coefficients from a regression model and turns them into a lexicon.

Argument and Default Value
==========================

Name of the lexicon to be created.

Details
=======

Use this switch to create a lexicon from a regression model. Either create the lexicon from a previously created model (using :doc:`fwflag_load_model`) or create a model using :doc:`fwflag_train_regression`. The name of the lexicon will be dd_ARGUMENT ; dd indicates the lexicon was data driven.

IMPORTANT: When creating the model, use :doc:`fwflag_no_standardize` or the model will not make any sense.

Only use regression algorithms that have linear coefficients (i.e by choosing the right :doc:`fwflag_model`), because the lexicon extraction equation won't make sense otherwise. This functionality hasn't totally been validated with advanced feature selection, so beware.
Also note that the coefficients won't be efficient to distinguish what features best characterize the outcomes looked at, use :doc:`fwflag_correlate` or other univariate techniques to get at that type of insight.


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` :doc:`fwflag_no_standardize` Needs one of these two switches:
:doc:`fwflag_train_regression` :doc:`fwflag_load_model` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Trains a regression model to predict age for users from 1grams, without standardizing
 # Will save the model to a picklefile called deleteMe.pickle, and create a lexicon called dd_testAgeLex
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age :doc:`fwflag_train_regression` :doc:`fwflag_save_model` :doc:`fwflag_picklefile` deleteMe.pickle 
 :doc:`fwflag_no_standardize` :doc:`fwflag_regression_to_lexicon` testAgeLex

 # Given a model that was previously made, this turns the model into a lexicon called dd_testAgeLex
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_load_model` :doc:`fwflag_picklefile` deleteMe.pickle 
 :doc:`fwflag_regression_to_lexicon` testAgeLex
References

Sap et al. (2014) - Developing Age and Gender Predictive Lexica over Social Media