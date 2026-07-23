.. _fwflag_no_standardize:
================
--no_standardize
================
Switch
======

--no_standardize

Description
===========

Disables column wise z-scoring of outcomes/features for regression/classification.

Argument and Default Value
==========================

False (i.e. standardizing is the default)

Details
=======

Usually, every outcome is z:doc:`fwflag_scored`, and so are the group_norms for every feature, but this switch disables that. This can sometimes improve prediction performance though usually it's slightly worse than with standardizing.

:doc:`fwflag_regression_to_lexicon`, :doc:`fwflag_classification_to_lexicon` need this flag.


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` 
Won't do anything without any of these switches:
:doc:`fwflag_train_regression`, :doc:`fwflag_combo_test_regression`, etc.
:doc:`fwflag_train_classifiers`, :doc:`fwflag_combo_test_classifiers`, etc.
:doc:`fwflag_regression_to_lexicon` :doc:`fwflag_classification_to_lexicon` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Trains a regression model to predict age for users from 1grams, without standardizing
 # Will save the model to a picklefile called deleteMe.pickle, and create a lexicon called testAgeLex
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age :doc:`fwflag_train_regression` :doc:`fwflag_save_model` :doc:`fwflag_picklefile` deleteMe.pickle 
 :doc:`fwflag_no_standardize` :doc:`fwflag_regression_to_lexicon` testAgeLex