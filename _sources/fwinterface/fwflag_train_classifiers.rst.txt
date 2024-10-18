.. _fwflag_train_classifiers:
===================
--train_classifiers
===================
Switch
======

--train_classifiers

Description
===========

Trains a classification model using the features given.

Argument and Default Value
==========================

None

Details
=======

This switch will cause the infrastructure to train a machine learning model to predict the outcome(s) (:doc:`fwflag_outcomes`) from the features in the feature tables :doc:`fwflag_f` (Note that you can put multiple feature tables in there). 
Features are loaded into memory, and are filtered/clustered using the feature selection (see below) and then standardized over the groups (unless :doc:`fwflag_no_standardize` is used), then fed into the classification model.
It is usually useful to use this switch with :doc:`fwflag_save_model`, but put the order of the features into the name cause those aren't yet stored in the model.

Feature Selection
In order to avoid overfitting, we have a couple of feature selection steps that one can do. Most of our feature selection is done using the Scikit:doc:`fwflag_Learn` package. To use it, we have a couple of pre:doc:`fwflag_made` feature selections, so just (un)comment the lines below this line:
 # feature selection:
 featureSelectionString = None
Every feature selector string will create an object if evaluated, and said object needs to have the following two functions:
fit(X, y)
transform(X)
If putting a lot of features into the model, it's good to use the pipeline feature selection:
 featureSelectionString = 'Pipeline([("1_mean_value_filter", OccurrenceThreshold(threshold=(X.shape[0]/100.0))), 
                                     ("2_univariate_select", SelectFwe(f_regression, alpha=70.0)), 
                                     ("3_rpca", RandomizedPCA(n_components=.4/len(self.featureGetters), random_state=42,
                                       whiten=False, iterated_power=3, max_components=X.shape[0]/max(1.5, len(self.featureGetters))))])'
If there aren't many features, you can choose not to use any feature selection. Talk to a CS PostDoc about this :)

Model selection
See below for choosing the model. Once the model is chosen, you should tweak the parameters by commenting in/out the appropriate line in classifyPredictor.py below
 # Model Parameters
 cvParams = {...
You can choose your model using :doc:`fwflag_model`, and choose one of the following:
svc (Support Vector Classification)
linear:doc:`fwflag_svc` (Support Vector Classification with Linear Kernel)
lr (Logistic Regression)
etc (ExtraTrees Classification)
rfc (RandomForrest Classification)
pac (Passive Agressive Classification)
lda (Linear Discriminant Analysis)

Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` 
Optional Switches:
:doc:`fwflag_group_freq_thresh` :doc:`fwflag_model` :doc:`fwflag_save_model` :doc:`fwflag_picklefile` :doc:`fwflag_no_standardize` :doc:`fwflag_sparse` :doc:`fwflag_classification_to_lexicon` etc.

Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Trains a classifier to predict the gender (a binary variable) for users from 1grams
 # Will save the model to a picklefile called deleteMeGender.pickle
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` gender :doc:`fwflag_train_classifiers` :doc:`fwflag_save_model` 
 :doc:`fwflag_picklefile` deleteMeGender.pickle
