.. _tut_pickle_build:
=======================
Building A Pickle Model
=======================

In this tutorial you will learn how to build a pickled regression model which can be used to predict values on a new data set. Specifically we will build an age language model.  It assumes you have completed the following tutorials:

* :doc:`tut_dla` 
* :doc:`tut_feat_tables` 
* :doc:`tut_pred` 


Commands
========

To create a pickle model we have two separate commands for both regression and classification tasks.  

Regression:

* :doc:`../fwinterface/fwflag_combo_test_regression`
* :doc:`../fwinterface/fwflag_train_regression`

Classification

* :doc:`../fwinterface/fwflag_combo_test_classifiers`
* :doc:`../fwinterface/fwflag_train_classifiers`

You will also use the follow, which are the same for both regression and classification:

* :doc:`../fwinterface/fwflag_save_model`
* :doc:`../fwinterface/fwflag_picklefile`

Training A Model
================

Finding the best model takes a few iterations as there are a few knobs to tune. We need to decide which feature set, model and feature selection pipeline to use. All of these choices depend on your level of analysis (document, user or community) as well. We will try a few options and use N-fold cross validation to determine the best model. N-fold cross validation is run with the following flags:

* :doc:`../fwinterface/fwflag_combo_test_regression`
* :doc:`../fwinterface/fwflag_folds`

First Pass
----------

As a first pass we will use 2000 LDA Facebook topics (available at wwbp.org) with the *ridgefirstpasscv* model. In *regressionPredictor.py* you can find the parameters for the *ridgefirstpasscv* model:

.. code-block:: python

	'ridgefirstpasscv': [
	            {'alphas': np.array([1, .01, .0001, 100, 10000, 1000000])}, 
	            ],

This specifies a range of Ridge Regression penalization parameters *alpha* and uses the sci-kit learn RidgeCV class. Note that we start with 1, iterate through smaller values and end with larger values. You will notice this same style when looking at other models in *regressionPredictor.py*. When the RidgeCV class finds two penalities with the same results it will default to the first parameter. By setting the first parameter to 1 we ensure that we will not default to a boundary case. 

The full call to *dlatkInterface* is as follows: 

.. code-block:: bash

	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' --outcome_table blog_outcomes --outcomes age --combo_test_regression --folds 10 --model ridgefirstpasscv

We run 10-fold cross validation, using the 2000 Facebook topics and consider only users with 500 or more words. The final output should look something like: 

.. code-block:: bash

	[TEST COMPLETE]

	{'age': {(): {1: {'N': 978,
          'R': nan,
          'R2': -0.88992485335879401,
          'R2_folds': -0.96270830361273707,
          'mae': 8.5336986508333688,
          'mae_folds': 8.5281778935127832,
          'mse': 130.66895432831001,
          'mse_folds': 130.30072681531286,
          'num_features': 2000,
          'r': 0.40222682541212029,
          'r_folds': 0.40305582938481005,
          'r_p': 2.4971926509824059e-39,
          'r_p_folds': 0.0012968155250152563,
          'rho': 0.45284615270670103,
          'rho_p': 1.288374195668318e-50,
          'se_R2_folds': 0.14099512415343221,
          'se_mae_folds': 0.16581363417963907,
          'se_mse_folds': 7.7572038557968686,
          'se_r_folds': 0.026510802106572903,
          'se_r_p_folds': 0.00086305723986972909,
          'se_train_mean_mae_folds': 0.20067386714575627,
          'test_size': 105,
          'train_mean_mae': 8.8679725446432318,
          'train_mean_mae_folds': 6.4643151489350172,
          'train_size': 873,
          '{modelFS_desc}': 'None',
          '{model_desc}': 'RidgeCV(alphas=array([ 1.00000e+00,  '
                          '1.00000e-02,  1.00000e-04,  '
                          '1.00000e+02,     1.00000e+04,  '
                          '1.00000e+06]),   cv=None, '
                          'fit_intercept=True, gcv_mode=None, '
                          'normalize=False,   scoring=None, '
                          'store_cv_values=False)'}}}}
	--
	Interface Runtime: 98.57 seconds
	DLATK exits with success! A good day indeed  ¯\_(ツ)_/¯.

Here we see a Pearson *r* = 0.402 but we also see *R* = nan so something is not right. Note that in the above output we have *N*=978 and *num_features*=2000. Since the number of features is twice the size of our observations one might consider using some feature selection. You can set the feature selection with either of the following flags:

* :doc:`../fwinterface/fwflag_feature_selection`
* :doc:`../fwinterface/fwflag_feature_selection_string`

We will use our *magic_sauce* pipeline which uses a pipeline of univariate feature selection (with a family wise error rate of 60) and Randomized PCA:

.. code-block:: python

	Pipeline([("1_mean_value_filter", OccurrenceThreshold(threshold=(X.shape[0]/100.0))), ("2_univariate_select", SelectFwe(f_regression, alpha=60.0)), ("3_rpca", RandomizedPCA(n_components=max(int(X.shape[0]/max(1.5,len(self.featureGetters))), min(50, X.shape[1])), random_state=42, whiten=False, iterated_power=3))])

We rerun the first command with the addition of the :doc:`../fwinterface/fwflag_feature_selection` flag:

.. code-block:: bash

	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' --outcome_table blog_outcomes --outcomes age --combo_test_regression --folds 10 --model ridgefirstpass --feature_selection magic_sauce

which gives us the following output:

.. code-block:: bash

	[TEST COMPLETE]  

	{'age': {(): {1: {'N': 978,
          'R': 0.61078813837532941,
          'R2': 0.37306214998000053,
          'R2_folds': 0.36172845861240155,
          'mae': 4.884718059241858,
          'mae_folds': 4.882525635420599,
          'mse': 43.346333662611386,
          'mse_folds': 43.319316495428431,
          'num_features': 2000,
          'r': 0.6225268232718818,
          'r_folds': 0.62849684211444135,
          'r_p': 5.0851259142747209e-106,
          'r_p_folds': 5.3572426774038191e-10,
          'rho': 0.66881845157843556,
          'rho_p': 8.1125738610915348e-128,
          'se_R2_folds': 0.019350854084730935,
          'se_mae_folds': 0.15951572834767661,
          'se_mse_folds': 2.6529378160101356,
          'se_r_folds': 0.019365962321772998,
          'se_r_p_folds': 2.8069174191858989e-10,
          'se_train_mean_mae_folds': 0.20067386714575627,
          'test_size': 105,
          'train_mean_mae': 3.3734409296293868,
          'train_mean_mae_folds': 6.4643151489350172,
          'train_size': 873,
          '{modelFS_desc}': "Pipeline(steps=[('1_mean_value_filter', "
                            'OccurrenceThreshold(threshold=2954)), '
                            "('2_univariate_select', "
                            'SelectFwe(alpha=60.0, '
                            'score_func=<function f_regression at '
                            "0x7fd8667328c8>)), ('3_rpca', "
                            'RandomizedPCA(copy=True, '
                            'iterated_power=3, max_components=None,    '
                            'n_components=793, random_state=42, '
                            'whiten=False))])',
          '{model_desc}': 'RidgeCV(alphas=array([ 1.00000e+00,  '
                          '1.00000e-02,  1.00000e-04,  '
                          '1.00000e+02,     1.00000e+04,  '
                          '1.00000e+06]),   cv=None, '
                          'fit_intercept=True, gcv_mode=None, '
                          'normalize=False,   scoring=None, '
                          'store_cv_values=False)'}}}}
	--
	Interface Runtime: 114.71 seconds
	DLATK exits with success! A good day indeed  ¯\_(ツ)_/¯.

Now we see both *r* and *R* are roughly equal. This gives us a good baseline to iterate on. 

Second Pass
-----------

Besides the final output of the various metrics you should also scroll up and see the output for each fold and look at the chosen *alpha* parameter:

.. code-block:: bash

	Fold 7
	   (feature group: 0): [Initial size: 978]
	[Train size: 881    Test size: 97]
	[Applying StandardScaler to X[0]: StandardScaler(copy=True, with_mean=True, with_std=True)]
	 X[0]: (N, features): (881, 2000)
	[Applying Feature Selection to X: Pipeline(steps=[('1_mean_value_filter', OccurrenceThreshold(threshold=2968)), ('2_univariate_select', Selec
	tFwe(alpha=60.0, score_func=<function f_regression at 0x7fd8667328c8>)), ('3_rpca', RandomizedPCA(copy=True, iterated_power=3, max_components
	=None,
	       n_components=800, random_state=42, whiten=False))])]
	SET THRESHOLD 2968
	 after feature selection: (N, features): (881, 800)
	[Training regression model: ridgefirstpasscv]
	model: RidgeCV(alphas=array([  1.00000e+00,   1.00000e-02,   1.00000e-04,   1.00000e+02,
	         1.00000e+04,   1.00000e+06]),
	    cv=None, fit_intercept=True, gcv_mode=None, normalize=False,
	    scoring=None, store_cv_values=False)
	  selected alpha: 10000.000000


For each fold the RidgeCV class chose *alpha* = 10000 so for our next pass we will choose the *ridgehighcv* model:

.. code-block:: python

	'ridgehighcv': [
	    {'alphas': np.array([10,100, 1, 1000, 10000, 100000, 1000000])}, 
	],

The command then becomes

.. code-block:: bash

	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' --outcome_table blog_outcomes --outcomes age --combo_test_regression --folds 10 --model ridgehighcv --feature_selection magic_sauce

which gives the output: 

.. code-block:: bash

	{'age': {(): {1: {'N': 978,
	      'R': 0.6279876727085858,
	      'R2': 0.39436851707394593,
	      'R2_folds': 0.37710881124945012,
	      'mae': 4.8017728082382849,
	      'mae_folds': 4.7992072704676758,
	      'mse': 41.873216515253354,
	      'mse_folds': 41.857939340623183,
	      'num_features': 2000,
	      'r': 0.63275749683012394,
	      'r_folds': 0.63739741140546302,
	      'r_p': 1.614437770178403e-110,
	      'r_p_folds': 5.732582650282395e-10,
	      'rho': 0.68505102312465327,
	      'rho_p': 1.991911281360676e-136,
	      'se_R2_folds': 0.028505009510342232,
	      'se_mae_folds': 0.156478519231599,
	      'se_mse_folds': 2.2790368274979054,
	      'se_r_folds': 0.018905676945535039,
	      'se_r_p_folds': 4.232561097638729e-10,
	      'se_train_mean_mae_folds': 0.20067386714575627,
	      'test_size': 105,
	      'train_mean_mae': 4.7403932140338458,
	      'train_mean_mae_folds': 6.4643151489350172,
	      'train_size': 873,
	      '{modelFS_desc}': "Pipeline(steps=[('1_mean_value_filter', "
	                        'OccurrenceThreshold(threshold=2954)), '
	                        "('2_univariate_select', "
	                        'SelectFwe(alpha=60.0, '
	                        'score_func=<function f_regression at '
	                        "0x7f0f75de38c8>)), ('3_rpca', "
	                        'RandomizedPCA(copy=True, '
	                        'iterated_power=3, max_components=None,    '
	                        'n_components=793, random_state=42, '
	                        'whiten=False))])',
	      '{model_desc}': 'RidgeCV(alphas=array([   10,   100,    1,  '
	                      '1000,  10000, 100000, 1000000]),   cv=None, '
	                      'fit_intercept=True, gcv_mode=None, '
	                      'normalize=False,   scoring=None, '
	                      'store_cv_values=False)'}}}}

This time *alpha* = 1000 is chosen for most of the folds and we see a jump in Pearson r. If we wanted to set this value for good we could switch the model to *ridge1000* but as we are going to try a few more iterations we will keep the *ridgehighcv* model.

We see a slight bump in Pearson r and a slight decrease in MSE. It is unknown if these are statistically significant differences but we will proceed as if they are. 

Third Pass
----------

In this next pass we will add additional feature sets. The 2000 Facebook topics are the best place to start but additional information can be found in the 1to3gram tables. We will add both a relative frequecy table (16to16) and a boolean table (16to1). We will also make sure these tables have rare words removed so as to not use too many features and overfit the model. Details on how to create these tables can be found in :doc:`tut_adv_fe`. Choosing the correct number and type of features is outside the scope of this tutorial. 

.. code-block:: bash

	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' 'feat$1to3gram$msgs$user_id$16to16$0_1' 'feat$1to3gram$msgs$user_id$16to1$0_1' --outcome_table blog_outcomes --outcomes age --combo_test_regression --folds 10 --model ridgehighcv --feature_selection magic_sauce

Before looking at the output you should note the order of your feature tables. This order is *VERY* important when you go to build your final model, which is explained in the next section.

The output should look like:

.. code-block:: bash

	{'age': {(): {1: {'N': 978,
	      'R': 0.65328970316714685,
	      'R2': 0.42678743626421878,
	      'R2_folds': 0.41097464037854003,
	      'mae': 4.7327336968238978,
	      'mae_folds': 4.7325620923091893,
	      'mse': 39.631780162099723,
	      'mse_folds': 39.631679282044317,
	      'num_features': 18394,
	      'r': 0.65400913034708874,
	      'r_folds': 0.66135842980913795,
	      'r_p': 1.9997143428258404e-120,
	      'r_p_folds': 1.194981277707842e-11,
	      'rho': 0.69474789987194996,
	      'rho_p': 7.612360070325049e-142,
	      'se_R2_folds': 0.020842882578666778,
	      'se_mae_folds': 0.11834531391511936,
	      'se_mse_folds': 1.9002353374669132,
	      'se_r_folds': 0.014106368216269394,
	      'se_r_p_folds': 9.4034265034747871e-12,
	      'se_train_mean_mae_folds': 0.20067386714575627,
	      'test_size': 105,
	      'train_mean_mae': 4.6141474094178232,
	      'train_mean_mae_folds': 6.4643151489350172,
	      'train_size': 873,
	      '{modelFS_desc}': "Pipeline(steps=[('1_mean_value_filter', "
	                        'OccurrenceThreshold(threshold=2954)), '
	                        "('2_univariate_select', "
	                        'SelectFwe(alpha=60.0, '
	                        'score_func=<function f_regression at '
	                        "0x7fa9b599e8c8>)), ('3_rpca', "
	                        'RandomizedPCA(copy=True, '
	                        'iterated_power=3, max_components=None,    '
	                        'n_components=281, random_state=42, '
	                        'whiten=False))])',
	      '{model_desc}': 'RidgeCV(alphas=array([   10,   100,    1,  '
	                      '1000,  10000, 100000, 1000000]),   cv=None, '
	                      'fit_intercept=True, gcv_mode=None, '
	                      'normalize=False,   scoring=None, '
	                      'store_cv_values=False)'}}}}

Again we see *r* and *R* fairly close to each other and also see a nice bump in performance over the topics alone model (r=0.63 to 0.65). 

We should go back and iterate on different feature selection pipelines, models (we have only tried Ridge regression but DLATK supports Lasso, Linear Regression, ElasticNet, Extra Trees, etc.) and model parameters but often you can get a "good enough" model without a full search over all models / parameters / features. 


Building The Final Model
========================

Finally, we want to build a model over the entire data set and save the model so we can apply to other data sets where age might not be available. To do that we use the following flag to train the model:

* :doc:`../fwinterface/fwflag_train_regression`

and the following to save the model:

* :doc:`../fwinterface/fwflag_save_model`
* :doc:`../fwinterface/fwflag_picklefile`

Our final command looks like:

.. code-block:: bash

	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' 'feat$1to3gram$msgs$user_id$16to16$0_1' 'feat$1to3gram$msgs$user_id$16to1$0_1' --outcome_table blog_outcomes --outcomes age --train_regression --model ridge1000 --feature_selection magic_sauce --save_model --picklefile ~/age.2000fbtopics.1to3grams.16to16.16to1.ridge1000.magic_sauce.gft500.pickle

which gives the output:

.. code-block:: bash

	[TRAINING COMPLETE]

	[Saving /home/user/age.2000fbtopics.1to3grams.16to16.16to1.ridge1000.magic_sauce.gft500.pickle]
	--
	Interface Runtime: 41.60 seconds
	DLATK exits with success! A good day indeed  ¯\_(ツ)_/¯.


Note the long file name. This tells us we built a model for age using 2000 Facebook topics, relative frequency (16to16) 1to3grams, boolean (16to1) 1to3grams using a Ridge regression (with penalization parameter = 1000), the "magic sauce" feature selection pipeline and that we limited the analysis to people with 500 or more words. This seems like a lot but you will most likely forget what went into your model and it's best to be explicit. 

Also it is *VERY* important to list the feature tables in the same exact order as you trained the model. When you go to apply the model to new data you will need to list the feature tables in the same exact order. 

To see how to apply your saved model to new data see the next tutorial: :doc:`tut_pickle_apply` 