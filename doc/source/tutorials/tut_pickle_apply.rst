.. _tut_pickle_apply:
=======================
Applying A Pickle Model
=======================

In this tutorial you will learn how to apply a pickled regression model to a new data set. It assumes you have completed the following tutorials:

* :doc:`tut_dla` 
* :doc:`tut_feat_tables` 
* :doc:`tut_pred` 

We use the model built in:

* :doc:`tut_pickle_build` 

Commands
========

To use a pickle model we have two separate commands for both regression and classification tasks. The 

Regression:

* :doc:`../fwinterface/fwflag_predict_regression_to_feats`
* :doc:`../fwinterface/fwflag_predict_regression_to_outcome_table`

Classification

* :doc:`../fwinterface/fwflag_predict_classifiers_to_feats`
* :doc:`../fwinterface/fwflag_predict_probabilities_to_feats`
* :doc:`../fwinterface/fwflag_predict_classifiers_to_outcome_table`


Predict Regression To Feature Table
===================================

This command will produce a "feature table" type table in MySQL where each row contains a group id, feature, value and group norm. This type of table is useful if you want to use these prediction results as features in a separate prediction task. 

Sample Command
--------------

.. code-block:: bash
	
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' \
	'feat$1to3gram$msgs$user_id$16to16' 'feat$1to3gram$msgs$user_id$16to1' --outcome_table ageDummy \
	--outcomes age --predict_regression_to_feat lbp_age  --load --picklefile \
	~/age.2000fbtopics.1to3grams.16to16.16to1.ridge1000.magic_sauce.gft500.pickle

Required Switches
-----------------

* :doc:`../fwinterface/fwflag_d`
* :doc:`../fwinterface/fwflag_t`
* :doc:`../fwinterface/fwflag_c`
* :doc:`../fwinterface/fwflag_f`
* :doc:`../fwinterface/fwflag_group_freq_thresh`
* :doc:`../fwinterface/fwflag_outcome_table`: 
* :doc:`../fwinterface/fwflag_outcomes`: 
* :doc:`../fwinterface/fwflag_predict_regression_to_feats`
* :doc:`../fwinterface/fwflag_load_model`
* :doc:`../fwinterface/fwflag_picklefile`

Feature Tables
--------------

The feature tables listed after the -f flag need to be in the same order as when the pickle file was created. This is usually specified in the name of the pickle. In this example the pickle name is age.2000fbtopics.1to3grams.16to16.16to1.ridge1000.magic_sauce.gft500.pickle which tells us that we need "2000fbtopics" (assumed to be the 2000 Facebook topics) and 1to3grams. We also see "16to16.16to1" in the name which tells us we need a 16to16 and 16to1 (boolean encoded) table.

The final tables should be:

* 2000 Facebook topics: feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16
* 1to3grams: feat$1to3gram$msgs$user_id$16to16$0_005
* Boolean 1to3grams: feat$1to3gram$msgs$user_id$16to1$0_005

Again, these tables need to be listed in this order

.. code-block:: bash

	-f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' 'feat$1to3gram$msgs$user_id$16to16$0_005'' feat$1to3gram$msgs$user_id$16to1$0_005'


Outcome Table
-------------

You must create a dummy outcome table as described in predict_regression_to_feats. The outcome table must have

* row for every outcome group_id (in this example, every message_id)
* a non-null value for every outcome listed after --outcomes (in this example, ope con ext agr neu)

You can do this in MySQL with:

.. code-block:: mysql

	mysql> CREATE TABLE ageDummy SELECT distinct user_id, 0 as age FROM msgs;
 	mysql> CREATE INDEX uidx on ageDummy (user_id);

	mysql> select * from ageDummy limit 5;
	+--------------------+------+
	| message_id         | age  |
	+--------------------+------+
	| 626654933077618688 |    0 |
	| 626654998773014528 |    0 |
	| 626655093023211520 |    0 |
	| 626655195976568832 |    0 |
	| 626655248321482752 |    0 |
	+--------------------+------+

Output Feature table
--------------------

The predicted values will be written to a feature table with the following name format: feat$p_modelType_ARGUMENT$message_table$group_id. The output feature table is as follows:

.. code-block:: mysql

	mysql> select * from feat$p_ridg_lbp_age$msgs$user_id order by group_id limit 2;
	+------+----------+------+---------------+---------------+
	| id   | group_id | feat | value         | group_norm    |
	+------+----------+------+---------------+---------------+
	|  793 |    28451 | age  | 36.5453007746 | 36.5453007746 |
	|  729 |   174357 | age  | 23.2885389879 | 23.2885389879 |
	+------+----------+------+---------------+---------------+


Predict Regression To Output Table
==================================

This command will produce an "outcome table" type table in MySQL where each row contains a group id and values for each outcome in the pickle model. This type of table is useful if you want to use these prediction results as outcomes in a separate DLA type task, for example using personality as controls when running DLA over some other outcome. 

Sample Command
--------------

.. code-block:: bash

	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' \
	'feat$1to3gram$msgs$user_id$16to16' 'feat$1to3gram$msgs$user_id$16to1'  --predict_regression_to_outcome_table lbp_age --load --picklefile \ 
	~/age.2000fbtopics.1to3grams.16to16.16to1.ridge1000.magic_sauce.gft500.pickle


Required Switches
-----------------

* :doc:`../fwinterface/fwflag_d`
* :doc:`../fwinterface/fwflag_t`
* :doc:`../fwinterface/fwflag_c`
* :doc:`../fwinterface/fwflag_f`
* :doc:`../fwinterface/fwflag_group_freq_thresh`
* :doc:`../fwinterface/fwflag_predict_regression_to_outcome_table`
* :doc:`../fwinterface/fwflag_load_model`
* :doc:`../fwinterface/fwflag_picklefile`

Feature Table
-------------

The feature tables listed after the -f flag need to be in the same order as when the pickle file was created. This is usually specified in the name of the pickle. In this example the pickle name is ocean.topics_grams16and1.pickle which tells us that we need topics (assumed to be the 2000 Facebook topics) and grams (assumed to be 1-3 grams). We also see 16and1 in the name which tells us we need a 16to16 and 16to1 (boolean encoded) table.

The final tables should be:

* 2000 Facebook topics: feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16
* 1to3grams: feat$1to3gram$msgs$user_id$16to16
* Boolean 1to3grams: feat$1to3gram$msgs$user_id$16to1

Again, these tables need to be listed in this order

.. code-block:: bash

	-f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' 'feat$1to3gram$msgs$user_id$16to16'' feat$1to3gram$msgs$user_id$16to1'

Output table
------------

The table created will look like: p_modelType$ARGUMENT If you used ridge with the argument in the sample command, for instance, it will look like p_ridg$lbp_age. Here "lbp" stands for "language based predictions".

The output table is in dense form and looks like:

.. code-block:: mysql

	mysql> select * from p_ridg$lbp_age limit 2;
	+---------+---------------+
	| user_id | age           |
	+---------+---------------+
	|   28451 | 36.5453007746 |
	|  174357 | 23.2885389879 |
	+---------+---------------+
	


