.. _fwflag_predict_regression:
====================
--predict_regression
====================
Switch
======

--predict_regression

Description
===========

Predicts the outcomes and compares the predicted values to the actual ones.

Argument and Default Value
==========================

None

Details
=======

Given a model (:doc:`fwflag_load_model`), this switch will predict the outcomes on the groups given in the outcome table and compare them to the actual values in the outcome table.
Make sure the feature tables are in the same order as they were when the model was created.

The output (printed to stdout) will contain the following values:
R^2 Coefficient of determination
R Square root of R^2
Pearson r Correlation between the predicted values and the original values (p:doc:`fwflag_value` in parentheses).
Spearman rho Rank correlation between the predicted values and the original values (p:doc:`fwflag_value` in parentheses).
Mean Squared Error Mean Squared Error
Mean Absolute Error Mean Absolute Error

Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` :doc:`fwflag_load_model` and :doc:`fwflag_picklefile` Optional Switches:
:doc:`fwflag_regression_to_lexicon` :doc:`fwflag_group_freq_thresh` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Loads the regression model in deleteMe.pickle, and uses the features to predict the ages of the users in 
 # masterstats_andy_r10k, and compares the predicted ages to the actual ages in the table.
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age :doc:`fwflag_load_model` :doc:`fwflag_picklefile` deleteMe.pickle 
 :doc:`fwflag_predict_regression` 