.. _fwflag_predict_classifiers_to_feats:
==============================
--predict_classifiers_to_feats
==============================
Switch
======

--predict_probabilities_to_feats

Description
===========

Produces probabilites for each outcome class and writes the probability of the max encoded class into a SQL table.

Argument and Default Value
==========================

Feature name for the SQL table.

Details
=======

Given a classification model (:doc:`fwflag_load_model`), this switch will predict probabilities on the groups given in the outcome table, for each class, and puts the probability of the max encoded class into a MySQL table. This is useful for a set of groups that you don't have the outcomes for, but you have a prediction model for it.

The table created will look like:

feat$p_modelType_ARGUMENT$message_table$group_id 

where modelType is the first 4 letter of the model name. If you used rfc for instance, it will look like 

feat$p_rfc_ARGUMENT$message_table$group_id.

Make sure the features are in the right order (i.e. the order they were put into when creating the model). A good place to check for that is the name of the pickle file (if you're using a pre:doc:`fwflag_made` picklefile, like those in here)

You need to make an output table that contains non null values for the outcomes & groups that you want probabilities for, cause it uses the :doc:`fwflag_predict_classifiers` code to run this, which is why it also outputs comparisons between the values in the outcome table and the predicted outcomes.

See :doc:`../tutorials/tut_pickle_apply` for more details on applying pickled models. 

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` 
* :doc:`fwflag_load_model` and :doc:`fwflag_picklefile` 


Example Commands
================


.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$1gra'  \
	--outcome_table blog_outcomes --outcomes genderDummy \
	--predict_probabilities_to_feats lbp_prob_gender  --load --picklefile \
	~/gender.2000fbtopics.lr.pickle


