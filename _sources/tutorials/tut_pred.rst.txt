.. _tut_pred:
=====================================================
Intro Prediction / Classification / Predictive Lexica
=====================================================

This is a continuation of the :doc:`tut_dla` tutorial.

STEP 3 - Prediction, Classification and Predictive Lexica
=========================================================

Prediction
----------
Next, we will try predicting age and gender with the above features. Since age is a continuous variable and gender binary we will use two different methods. We begin with predicting age. Here is the basic command for prediction:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id \ 
	-f 'feat$cat_met_a30_2000_cp_w$msgs_xxx$user_id$1gra' 'feat$1gram$msgs_xxx$user_id' \ 
	--outcome_table blog_outcomes  --group_freq_thresh 500 \ 
	--outcomes age --output_name xxx_age_output  \ 
	--nfold_test_regression --model ridgecv --folds 10 

Brief descriptions of the flags:

* :doc:`../fwinterface/fwflag_combo_test_regression`: Does K folds of splitting the data into test and training set, trains a model on training set and predicts it on the test set. K-fold cross validation 
* :doc:`../fwinterface/fwflag_model`: Model to use when predicting. For example: ridge regression, lasso regression, logistic regression
* :doc:`../fwinterface/fwflag_folds`: number of chunks to split the data into 
* :doc:`../fwinterface/fwflag_prediction_csv`: print prediction results to a csv (optional)

The above command will randomly split the data set into K chunks (or folds) where K is specified by --folds (default value is 5). For each chunk, a model is trained on the remaining K-1 chunks and tested on this chunk. Note that we are now using two feature tables (1grams and Facebook topics), as opposed to Step 2 above.

After each fold you will see the following output:

.. code-block:: bash

	model: RidgeCV(alphas=array([   1000,   10000,  100000, 1000000]), cv=None,
	  fit_intercept=True, gcv_mode=None, normalize=False, scoring=None,
	  store_cv_values=False)
	selected alpha: 1000.000000
	predict: applying standard scaler to X[0]: StandardScaler(copy=True, with_mean=True, with_std=True)
	predict: applying standard scaler to X[1]: StandardScaler(copy=True, with_mean=True, with_std=True)
	predict: combined X shape: (50, 67593)
	predict: regression intercept: 25.602222
	*FOLD R^2: 0.4065 (MSE: 41.8599; MAE: 4.7181; mean train mae: 6.7604)

Note that we are training a separate model for each fold and could have different alpha's for each model.

and then at the end of the K folds you will see:

.. code-block:: bash

	*Overall R^2:          0.3010
	*Overall FOLDS R^2:    0.3136 (+- 0.0224)
	*R (sqrt R^2):         0.5487
	*Pearson r:            0.6496 (p = 0.00000)
	*Folds Pearson r:      0.6814 (p = 0.00001)
	*Spearman rho:         0.7277 (p = 0.00000)
	*Mean Squared Error:   77.7278
	*Mean Absolute Error:  5.5103
	*Train_Mean MAE:       2.6129

which is the average score over all folds. 

Finally, once each outcome is finished, you will see the following output (only if --prediction_csv is NOT used):

.. code-block:: bash

	{'age': {(): {1: {'N': 499,
                        'R': 0.54865313166717689,
                        'R2': 0.3010202588882005,
                        'R2_folds': 0.3136392251334551,
                        'mae': 5.5102667175760027,
                        'mae_folds': 5.4962114905936383,
                        'mse': 77.727807054395043,
                        'mse_folds': 76.30786952879339,
                        'num_features': 67558,
                        'r': 0.64961384918228648,
                        'r_folds': 0.68141019974698003,
                        'r_p': 3.7941701177133611e-61,
                        'r_p_folds': 1.4329061542488414e-05,
                        'rho': 0.72770801319050482,
                        'rho_p': 2.034045921063826e-83,
                        'se_R2_folds': 0.022425595182743487,
                        'se_mae_folds': 0.31528063131662087,
                        'se_mse_folds': 13.616220914873225,
                        'se_r_folds': 0.024241974166569567,
                        'se_r_p_folds': 1.1915221687336874e-05,
                        'se_train_mean_mae_folds': 0.39617060533142334,
                        'test_size': 58,
                        'train_mean_mae': 2.6128928956045994,
                        'train_mean_mae_folds': 7.2689576980217385,
                        'train_size': 441,
                        '{modelFS_desc}': 'None',
                        '{model_desc}': 'RidgeCV(alphas=array([  1000,  10000, 100000, 1000000]), cv=None,   
                             fit_intercept=True, gcv_mode=None, normalize=False, scoring=None,   store_cv_values=False)'}}},

The prediction_csv command will produce a csv file called xxx_age_output.predicted_data.csv which includes prediction scores for each outcome and each group_id:

.. code-block:: bash

	Id,age__withLanguage
	e73b38988d4a277a1ac12c258fb33a14,25.71484278807899
	fb2eecbe942c268e0d47a377dde7831a,27.556797752649086
	2a4ea8a5ac157246feedfcf72edad5ff,25.42848778125374
	6a56f67c249e8e14403b3f40231cdde4,21.951236181408891
	e571dcc7fa1a6f1ebb1aba8d05d39b5c,23.937405646479284
	12508f45aa9c98ae6e6816b030b6b581,21.399554821560564

Classification
--------------
Next, we will predict gender. Since this is a binary outcome we switch to classification via logistic regression. Here is the command: 

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id \ 
	-f 'feat$cat_met_a30_2000_cp_w$msgs_xxx$user_id$1gra' 'feat$1gram$msgs_xxx$user_id$16to16' \ 
	--outcome_table blog_outcomes  --group_freq_thresh 500   \ 
	--outcomes gender --output_name xxx_gender_output \ 
	--nfold_test_classifiers --model lr --folds 10  --prediction_csv

Brief descriptions of the flags:

* :doc:`../fwinterface/fwflag_combo_test_classifiers`: Does K folds of splitting the data into test and training set, trains a model on training set and predicts it on the test set (K-fold cross validation).

At the end of each fold you will see the output:

.. code-block:: bash

	*confusion matrix: 
	[[17  2]
	[12 19]]

	*precision and recall: 
	         precision    recall  f1-score   support
	      0       0.59      0.89      0.71        19
	      1       0.90      0.61      0.73        31
	avg / total       0.78      0.72      0.72        50

	*FOLD ACC: 0.7200 (mfclass_acc: 0.6200); mfclass: 1; auc: 0.8336

At the end of the K folds you will see the output:

.. code-block:: bash 

 	{'gender': {(): {1: {'acc': 0.73199999999999998,
                           'auc': 0.84652080652080652,
                           'f1': 0.75985663082437271,
                           'folds_acc': 0.73199999999999998,
                           'folds_auc': 0.84338861809425469,
                           'folds_f1': 0.7577487943864093,
                           'folds_mfclass_acc': 0.63000000000000012,
                           'folds_precision': 0.87205025089102173,
                           'folds_recall': 0.67282259377235909,
                           'folds_rho': 0.47423319603178121,
                           'folds_rho-p': 0.0059028924590036405,
                           'folds_se_acc': 0.020823064135712589,
                           'folds_se_auc': 0.015851020617975636,
                           'folds_se_f1': 0.017977732950626615,
                           'folds_se_mfclass_acc': 0.03408812109811863,
                           'folds_se_precision': 0.023172378772080934,
                           'folds_se_recall': 0.01997606345512792,
                           'folds_se_rho': 0.041178583963528741,
                           'folds_se_rho-p': 0.0024991387176062363,
                           'mfclass': '1',
                           'mfclass_acc': 0.63,
                           'num_classes': '2',
                           'num_features': 67593,
                           'predictions': {u'003ae43fae340174a67ffbcf19da1549': 0,
                                            ...}
                           'test_size': 50,
                           'train_size': 450,
                           '{modelFS_desc}': "Pipeline(steps=[('1_univariate_select', 
                                 SelectFwe(alpha=30.0, score_func=<function f_classif at 0x7ff58b5bfcf8>)), 
                                 ('2_rpca', RandomizedPCA(copy=True, iterated_power=3, n_components=200, random_state=42,    
                                 whiten=False))])",
                           '{model_desc}': "LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True, 
                                 intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                 penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
                                 verbose=0, warm_start=False)"}}}}
              

Finally, the prediction_csv command will produce a csv file called xxx_age_output.predicted_data.csv which includes prediction scores for each outcome and each group_id:

.. code-block:: bash

	Id,gender__withLanguage
	e73b38988d4a277a1ac12c258fb33a14,0
	fb2eecbe942c268e0d47a377dde7831a,0
	2a4ea8a5ac157246feedfcf72edad5ff,1
	6a56f67c249e8e14403b3f40231cdde4,1
	e571dcc7fa1a6f1ebb1aba8d05d39b5c,0

Predictive Lexica
-----------------
In this step we will use one of our data driven lexica to make predictions from out text data. We do this the same way we applied the LIWC and Facebook topics in Step 1 via --add_lex_table. Since this is a weighted lexicon we must use the :doc:`../fwinterface/fwflag_weighted_lexicon` flag.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id --add_lex_table -l dd_emnlp14_ageGender --weighted_lexicon

This will create the feature table feat$cat_dd_emnlp14_ageGender_w$msgs_xxx$user_id$16to16. Looking at the first 10 entries of this table:

.. code-block:: mysql

	+----+----------------------------------+--------+-------+-------------------+
	| id | group_id                         | feat   | value | group_norm        |
	+----+----------------------------------+--------+-------+-------------------+
	|  1 | 003ae43fae340174a67ffbcf19da1549 | GENDER |  3414 |  2.31201123597329 |
	|  2 | 003ae43fae340174a67ffbcf19da1549 | AGE    |  3708 |  26.7488590924939 |
	|  3 | 01f6c25f87600f619e05767bf8942a5f | GENDER |  1224 | -4.27544591260491 |
	|  4 | 01f6c25f87600f619e05767bf8942a5f | AGE    |  1338 |  19.3132484926424 |
	|  5 | 02be98c1005c0e7605385fbc5009de61 | GENDER |  2327 |  1.00734567214878 |
	|  6 | 02be98c1005c0e7605385fbc5009de61 | AGE    |  2567 |  22.5155537183235 |
	|  7 | 0318cc38971845f7470f34704de7339d | GENDER |  2779 |  1.07764014307349 |
	|  8 | 0318cc38971845f7470f34704de7339d | AGE    |  2945 |  20.9815875967542 |
	|  9 | 040b2b154e4074a72d8a7b9697ec76d2 | GENDER |  5218 |  1.35815398645287 |
	| 10 | 040b2b154e4074a72d8a7b9697ec76d2 | AGE    |  5559 |  20.6623785068464 | 
	+----+----------------------------------+--------+-------+-------------------+

We see an age and gender score for each use. The group_norm column contains the predicted age and gender value. For the gender value we need only look at the sign of the group_norm. Here a positive values are female predictions and negative values are male predictions. To compare this to the outcome table we use the following MySQL command:

.. code-block:: mysql

	mysql> SELECT feat.group_id, feat.feat, feat.group_norm, outcomes.age, outcomes.gender 
	FROM dla_tutorial.blog_outcomes as outcomes
	INNER JOIN dla_tutorial.feat$cat_dd_emnlp14_ageGender_w$msgs_xxx$user_id$16to16 as feat 
	ON outcomes.user_id=feat.group_id limit 10;

which gives the output

.. code-block:: mysql

	+----------------------------------+--------+-------------------+-----------+--------------+
	| group_id                         | feat   | group_norm        | age       | gender   |
	+----------------------------------+--------+-------------------+-----------+--------------+
	| 33522bc535275457a87e20b3d0be71f2 | GENDER |  3.23204271891755 |        26 |            1 |
	| 33522bc535275457a87e20b3d0be71f2 | AGE    |  26.8702764242703 |        26 |            1 |
	| d388b5ca68ff9192bab2f6b53a6cab13 | GENDER | 0.992287422430842 |        50 |            1 |
	| d388b5ca68ff9192bab2f6b53a6cab13 | AGE    |  40.8649261269079 |        50 |            1 |
	| 8ad4098a61dfc3e42916a35293802a59 | GENDER | -2.15381597749128 |        18 |            0 |
	| 8ad4098a61dfc3e42916a35293802a59 | AGE    |  23.6505762607835 |        18 |            0 |
	| c35536977baa796cdf671697400f16ac | GENDER |  2.60813970285153 |        20 |            1 |
	| c35536977baa796cdf671697400f16ac | AGE    |  19.1539536735386 |        20 |            1 |
	| b67875fbdbabb1187715721697517139 | GENDER | -1.55682728995491 |        25 |            0 |
	| b67875fbdbabb1187715721697517139 | AGE    |  20.9407575669782 |        25 |            0 | 
	+----------------------------------+--------+-------------------+-----------+--------------+

One can also create a predictive lexicon using the :doc:`../fwinterface/fwflag_regression_to_lexicon` switch, which is beyond the scope of this tutorial. For more information please read `Sap et al. (2014) - Developing Age and Gender Predictive Lexica over Social Media <http://wwbp.org/papers/emnlp2014_developingLexica.pdf>`_.