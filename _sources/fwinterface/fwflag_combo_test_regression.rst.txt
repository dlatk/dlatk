.. _fwflag_combo_test_regression:
=======================
--nfold_test_regression
=======================
Switch
======

--nfold_test_regression or --combo_test_regression

Description
===========

Does K folds of splitting the data into test and training set, trains a model on training set and predicts it on the test set. K-fold cross validation.

Argument and Default Value
==========================

None

Details
=======

Similarly to :doc:`fwflag_test_regression`, this switch causes the data to be randomly spit in N chunks (where N is either 5 by default or defined by :doc:`fwflag_folds`). For each chunk, a model is trained on the remaining N-1 chunks and tested on this chunk.

After all chunks have been tested on, the accuracies and other metrics are averaged and printed out, which says something about the parameters and model chosen.

If you included :doc:`fwflag_controls` in your command, multiple K-fold CV will be done, one run for each of all the possible subsets of controls. Unless :doc:`fwflag_control_combo_sizes` or :doc:`fwflag_all_controls_only` is specified.

Note that all remarks about feature selection and model/parameter selection from :doc:`fwflag_train_regression` apply, so please read that section.

Output
------

Per fold, the following line will be printed to the standard output (numbers are examples)

.. code-block:: bash

 FOLD R^2: 0.6293 (MSE: 26.8697; MAE: 3.6475; mean train mae: 6.4276)

R^2 Coefficient of Determination
MSE Mean Squared Error of the prediction on the test set
MAE Mean Absolute Error of the prediction on the test set
mean train mae Mean Absolute Error of predicting the mean from the training set  on the test set (baseline)
At the end of the folds, you'll get output that looks like this:

.. code-block:: bash


  {'age': {(): {1: {'N': 9998,
                   'R': 0.79009985308749586,
                   'R2': 0.62425777784888248,
                   'R2_folds': 0.62368517193313822,
                   'mae': 3.7619989539884977,
                   'mae_folds': 3.762037681819328,
                   'mse': 29.898025697226569,
                   'mse_folds': 29.899256104306499,
                   'num_features': 10797,
                   'r': 0.79109013487672142,
                   'r_folds': 0.7909194802451609,
                   'r_p': 0.0,
                   'r_p_folds': 1.5067491373744111,                    
                   'rho': 0.77229872578756076,
                   'rho_p': 0.0,
                   'se_R2_folds': 0.0083296286255124322,
                   'se_mae_folds': 0.028974068999092516,
                   'se_mse_folds': 0.89544586136045368,
                   'se_r_folds': 0.0054909123600744899,
                   'se_r_p_folds': 0.0,
                   'se_train_mean_mae_folds': 0.041434284485758692,
                   'test_size': 1007,
                   'train_mean_mae': 5.2418219903436825,
                   'train_mean_mae_folds': 6.5176954815233685,
                   'train_size': 8991,
                   '{modelFS_desc}': "Pipeline(steps=[('1_mean_value_filter', OccurrenceThreshold(threshold=808381L)), 
                                                      ('2_univariate_select', SelectFwe(alpha=70.0, score_func=<function f_regression at 0x7fb5691946e0>)), 
                                                      ('3_rpca', RandomizedPCA(copy=True, iterated_power=3, max_components=5994.0,
                                                                                n_components=1655, random_state=42, whiten=False))])",
                   '{model_desc}': 'RidgeCV(alphas=array([ 1.00000e+03,  1.00000e+00,  1.00000e:doc:`fwflag_01`,  1.00000e+01,     1.00000e+02,  1.00000e+04,  1.00000e+05]),
                                            cv=None, fit_intercept=True, gcv_mode=None, loss_func=None,   normalize=False, 
                                            score_func=None, scoring=None, store_cv_values=False)'}}}} 

If there were controls included, you get 


.. code-block:: bash

  {'age': {(): {1: {'N': 9998,
                    ...}},
             ('gender',): {0: {'N': 9998,
                               ...},
                           1: {'N': 9998,
                               ...}}
  }}

The first set of metrics ((): {1...) is the prediction performance of the language features alone, without any of the controls.

('gender',) means gender was included as a control in the prediction of age, and the first item in the dictionary ({0: {...}) is the performance using just the control values, no language, and then the ({1: {...}) is the performance with both controls and language. As you add controls, there will be 2n result dictionaries.


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` 

Optional Switches:

* :doc:`fwflag_model`
* :doc:`fwflag_no_standardize`
* :doc:`fwflag_folds`
* :doc:`fwflag_sparse`
* :doc:`fwflag_group_freq_thresh`
* :doc:`fwflag_all_controls_only`
* :doc:`fwflag_control_combo_sizes`
* :doc:`fwflag_no_lang` 

Example Commands
================

.. code-block:: bash


  # Runs 10-fold cross validation on predicting the users ages from 1grams.
  # This essentially will tell you how well your model & features do at predicting age.
  dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$16to16' --outcome_table blog_outcomes \
  --outcomes age --combo_test_regression --model ridgecv --folds 10
