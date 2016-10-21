.. _fwflag_save_model:
============
--save_model
============
Switch
======

--save_model

Description
===========

Saves the model that you just trained.

Argument and Default Value
==========================

None

Details
=======

If you are running a command that creates/trains a model, using this will tell the infrastructure to save the model. This switch requires the use of :doc:`fwflag_picklefile`. After saving the model, you can use the file using :doc:`fwflag_load_model`. 
See one of these three for more info:
:doc:`fwflag_train_regression` :doc:`fwflag_train_classifiers` :doc:`fwflag_cca` 
Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` :doc:`fwflag_picklefile` Optional Switches:
:doc:`fwflag_train_regression` and anything optional to that
:doc:`fwflag_train_classifiers` and anything optional to that
:doc:`fwflag_cca` and anything optional to that
etc.

Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Trains a regression model to predict age for users from 1grams
 # Will save the model to a picklefile called deleteMe.pickle
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age :doc:`fwflag_train_regression` :doc:`fwflag_save_model` :doc:`fwflag_picklefile` deleteMe.pickle
