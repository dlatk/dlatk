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

* :doc:`fwflag_train_regression` :doc:`fwflag_train_classifiers` :doc:`fwflag_cca` 

It is important to name your descriptively name your file. For example, you should include the outcome and features the model was built on. When applying the model you must specify the feature tables in the same order as when the model was created, so if this is reflected in your file name then it will save you time in the future. 

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` 
* :doc:`fwflag_picklefile` 

Optional Switches:

* :doc:`fwflag_train_regression` 
* :doc:`fwflag_train_classifiers` 
* :doc:`fwflag_cca` 


Example Commands
================

.. code-block:: bash

	# Trains a regression model to predict age for users from 1grams
	# Will save the model to a picklefile called age.1grams16to16.pickle
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$16to16$0_01'  --outcome_table blog_outcomes --outcomes age --train_regression --save_model --picklefile age.1grams16to16.pickle


