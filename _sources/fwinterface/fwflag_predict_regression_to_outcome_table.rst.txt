.. _fwflag_predict_regression_to_outcome_table:
=====================================
--predict_regression_to_outcome_table
=====================================
Switch
======

--predict_regression_to_outcome_table

Description
===========

Predicts the outcomes and puts the predicted values into a SQL outcome table

Argument and Default Value
==========================

Name of the new outcome table

Details
=======

Given a model (:doc:`fwflag_load_model`), this switch will predict the outcomes on the groups given in the feature table and puts the values into a MySQL outcome table. This is useful for a set of groups that you don't have the outcomes for, but you have a prediction model for it (here are the outcomes we can predict from language).
The table created will look like:
p_modelType$ARGUMENT
If you used ridge for instance, it will look like p_ridg$ARGUMENT.

Make sure the features are in the right order (i.e. the order they were put into when creating the model).

The table will have one column for the correlation field (:doc:`fwflag_c`) and a column for each outcome in the model.


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f` :doc:`fwflag_load_model` and :doc:`fwflag_picklefile` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Loads the regression model in deleteMe.pickle, and uses the features to predict the ages of the users in 
 # feat$1gram$messages_en$user_id$16to16$0_01, and inserts those values into a table called p_ridg$deleteMe.
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_load_model` :doc:`fwflag_picklefile` deleteMe.pickle :doc:`fwflag_predict_regression_to_outcome_table` deleteMe
