.. _tut_pickle_apply:
=======================
Applying A Pickle Model
=======================

In this tutorial you will learn how to apply a pickled regression model to a new data set. It assumes you have completed the following tutorials:

* :doc:`tut_dla` 
* :doc:`tut_feat_tables` 
* :doc:`tut_pred` 

Commands
========

To use a pickle model we have two separate commands for both regression and classification tasks. The 

Regression:

* :doc:`../fwinterface/fwflag_predict_regression_to_feats`
* :doc:`../fwinterface/fwflag_predict_regression_to_outcome_table`

Classification

* :doc:`../fwinterface/fwflag_predict_classifiers_to_feats`
* :doc:`../fwinterface/fwflag_predict_classification_to_outcome_table`


Predict Regression To Feature Table
===================================

This command will produce a "feature table" type table in MySQL where each row contains a group id, feature, value and group norm. This type of table is useful if you want to use these prediction results as features in a separate prediction task. 

Sample Command
--------------

.. code-block:: bash

	./dlatkInterface.py -d dla_tutorial -t msgsEn_r500 -c user_id --group_freq_thresh 500 -f 'feat$cat_met_a30_2000_cp_w$msgsEn_r500$user_id$16to16' \
   'feat$1to3gram$msgsEn_r500$user_id$16to16' 'feat$1to3gram$msgsEn_r500$user_id$16to1' --outcome_table oceanDummy \
   --outcomes ope con ext agr neu --predict_regression_to_feat lbp_ocean  --load --picklefile \
   ~/pickles/ocean.topics_grams16and1.pickle

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

The feature tables listed after the -f flag need to be in the same order as when the pickle file was created. This is usually specified in the name of the pickle. In this example the pickle name is ocean.topics_grams16and1.pickle which tells us that we need topics (assumed to be the 2000 Facebook topics) and grams (assumed to be 1-3 grams). We also see 16and1 in the name which tells us we need a 16to16 and 16to1 (boolean encoded) table.

The final tables should be:

* 2000 Facebook topics: feat$cat_met_a30_2000_cp_w$statuses_er1$study_code$16to16
* 1to3grams: feat$1to3gram$statuses_er1$study_code$16to16$0_005
* Boolean 1to3grams: feat$1to3gram$statuses_er1$study_code$16to1$0_005

Again, these tables need to be listed in this order

.. code-block:: bash

	-f 'feat$cat_met_a30_2000_cp_w$statuses_er1$study_code$16to16' 'feat$1to3gram$statuses_er1$study_code$16to16$0_005'' feat$1to3gram$statuses_er1$study_code$16to1$0_005'


Outcome Table
-------------

You must create a dummy outcome table as described in predict_regression_to_feats. The outcome table must have

* row for every outcome group_id (in this example, every message_id)
* a non-null value for every outcome listed after --outcomes (in this example, ope con ext agr neu)

You can do this in MySQL with:

.. code-block:: mysql

	mysql> CREATE TABLE oceanDummy SELECT distinct user_id, 0 as ope, 0 as con, 0 as ext, 0 as agr, 0 as neu FROM msgsEn_r500;
 	mysql> CREATE INDEX uidx on oceanDummy (user_id);

	mysql> select * from oceanDummy limit 5;
	+--------------------+------+------+------+------+------+
	| message_id         | ope  | con  | ext  | agr  | neu  |
	+--------------------+------+------+------+------+------+
	| 626654933077618688 |    0 |    0 |    0 |    0 |    0 |
	| 626654998773014528 |    0 |    0 |    0 |    0 |    0 |
	| 626655093023211520 |    0 |    0 |    0 |    0 |    0 |
	| 626655195976568832 |    0 |    0 |    0 |    0 |    0 |
	| 626655248321482752 |    0 |    0 |    0 |    0 |    0 |
	+--------------------+------+------+------+------+------+

Output Feature table
--------------------

The predicted values will be written to a feature table with the following name format: feat$p_modelType_ARGUMENT$message_table$group_id. The output feature table is as follows:

.. code-block:: mysql

	mysql> select * from feat$p_ridg_lbp_ocean$msgsEn_r500$user_id order by group_id limit 10;
	+------+----------+------+---------------+---------------+
	| id   | group_id | feat | value         | group_norm    |
	+------+----------+------+---------------+---------------+
	|  793 |    28451 | con  | 3.54530077462 | 3.54530077462 |
	| 1702 |    28451 | agr  |  3.6534310168 |  3.6534310168 |
	| 2611 |    28451 | ope  | 3.84254078795 | 3.84254078795 |
	| 3520 |    28451 | neu  | 2.45199953382 | 2.45199953382 |
	| 4429 |    28451 | ext  | 3.68977021602 | 3.68977021602 |
	|  729 |   174357 | con  | 3.28853898719 | 3.28853898719 |
	| 1638 |   174357 | agr  | 3.44217927995 | 3.44217927995 |
	| 2547 |   174357 | ope  | 3.86768411414 | 3.86768411414 |
	| 3456 |   174357 | neu  | 3.33320166467 | 3.33320166467 |
	| 4365 |   174357 | ext  | 3.53265219915 | 3.53265219915 | 
	+------+----------+------+---------------+---------------+


Predict Regression To Output Table
==================================

This command will produce an "outcome table" type table in MySQL where each row contains a group id and values for each outcome in the pickle model. This type of table is useful if you want to use these prediction results as outcomes in a separate DLA type task, for example using personality as controls when running DLA over some other outcome. 

Sample Command
--------------

.. code-block:: bash

	./dlatkInterface.py -d dla_tutorial -t msgsEn_r500 -c user_id --group_freq_thresh 500 -f 'feat$cat_met_a30_2000_cp_w$msgsEn_r500$user_id$16to16' \
	'feat$1to3gram$msgsEn_r500$user_id$16to16' 'feat$1to3gram$msgsEn_r500$user_id$16to1'  --predict_regression_to_outcome_table lbp_ocean --load --picklefile \ 
	~/pickles/ocean.topics_grams16and1.pickle


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

* 2000 Facebook topics: feat$cat_met_a30_2000_cp_w$statuses_er1$study_code$16to16
* 1to3grams: feat$1to3gram$statuses_er1$study_code$16to16$0_005
* Boolean 1to3grams: feat$1to3gram$statuses_er1$study_code$16to1$0_005

Again, these tables need to be listed in this order

.. code-block:: bash

	-f 'feat$cat_met_a30_2000_cp_w$statuses_er1$study_code$16to16' 'feat$1to3gram$statuses_er1$study_code$16to16$0_005'' feat$1to3gram$statuses_er1$study_code$16to1$0_005'

Output table
------------

The table created will look like: p_modelType$ARGUMENT If you used ridge with the argument in the sample command, for instance, it will look like p_ridg$lbp_ocean.

The output table is in dense form and looks like:

.. code-block:: mysql

	mysql> select * from p_ridg$lbp_ocean limit 5;
	+----------------------------------+---------------------+-----------------------+----------------------+--------------------+--------------------+
	| user_id                          | agr                 | con                   | ext                  | neu                | ope                |
	+----------------------------------+---------------------+-----------------------+----------------------+--------------------+--------------------+
	| 003ae43fae340174a67ffbcf19da1549 |   -0.13095289515739 |     0.443488877877727 |    0.077908369739178 |  0.306820890920635 |   0.39442538949098 |
	| 01f6c25f87600f619e05767bf8942a5f |  -0.227043294966826 |    -0.201523681873899 |   -0.153953639046984 |  0.281840835050514 |  0.169342300429294 |
	| 02be98c1005c0e7605385fbc5009de61 |  -0.014249758237521 | -0.000907793806014523 | -0.00360407661522653 |  0.167192811796462 | -0.451018678659986 |
	| 0318cc38971845f7470f34704de7339d | -0.0217284226094123 |     -0.11319834497912 |   -0.190221082161925 |  -0.18154287233182 | -0.067660564250442 |
	| 040b2b154e4074a72d8a7b9697ec76d2 | -0.0456476898016594 |    -0.109844694383632 |    0.234373279991295 | 0.0821232707338745 |  0.494226569168025 |
	+----------------------------------+---------------------+-----------------------+----------------------+--------------------+--------------------+


