.. _fwflag_picklefile:
============
--picklefile
============
Switch
======

--picklefile

Description
===========

Switch to indicate name of pickle file to save/load a model from.

Argument and Default Value
==========================

Name of picklefile

Details
=======

If you are running a command that creates/trains or loads/uses a model, using this will tell the infrastructure where to find/put the model. This switch requires the use of :doc:`fwflag_load_model` or :doc:`fwflag_save_model`. 


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` :doc:`fwflag_save_model` or :doc:`fwflag_load_model` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Trains a regression model to predict age for users from 1grams
 # Will save the model to a picklefile called deleteMe.pickle
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age :doc:`fwflag_train_regression` :doc:`fwflag_save_model` :doc:`fwflag_picklefile` deleteMe.pickle

 # Uses the trained regression model (deleteMe.pickle) to predict age for users from 1grams
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age :doc:`fwflag_predict_regression` :doc:`fwflag_load_model` :doc:`fwflag_picklefile` deleteMe.pickle
