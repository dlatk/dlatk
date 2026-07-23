.. _tut_dla:
==================
DLA Rules of Thumb
==================

p = |feats|

First rule of thumb: there really are no hard and fast rules that always apply but these are best places to start.

DLA
---

* Feat_occ_filter (:doc:`../fwinterface/fwflag_feat_occ_filter`):

	* Rule of thumb: N/2 < p < N

* Depends on:

	* Expected effect sizes
	* Sparsity of your observations
	* How many words do you have per person? (measure how well we are estimating word use rates)
	* "True" strength of relationship

Colloc_filter
-------------

* See :doc:`../fwinterface/fwflag_feat_colloc_filter`
* When to apply?

	* good for DLA
	* usually not good for prediction (less accurate models)

* generally a pmi threshold of 3 works for anything from 2grams to 4grams. 

Prediction
----------

* Feat_occ_filter

	* See :doc:`../fwinterface/fwflag_feat_occ_filter`
	* Rule of thumb: N < p < 2*N 
	* (when you’re doing "magic sauce"”" feature selection or LASSO (L1) penalization)

* Colloc filter: doesn’t usually help (sometimes hurts)
* What usually works best 

	* Regression (listed in order to what usually works best)

		* Feat_occ_filter => Univariate selection => PCA => L2 (ridge) regression (:doc:`../fwinterface/fwflag_feature_selection` magic sauce -:doc:`../fwinterface/fwflag_model` ridgecv)
		* LASSO L1 regression (with no separate feature selection)
		* ElasticNet L1L2 regression (presumably worse because there is one more hyper-parameter to set)	

	* Choosing dimensions for PCA 
		
		* If a lot of observations (>>10k): 10% of p
		* If few observations (< 10k) 50% of N
		* (see more complicated funtions in regressionPredictor.py featureSelectionString)
	
	* Classification

		* L1 linear-svm (:doc:`../fwinterface/fwflag_model` linear-svc)
		* L1 logistic regression (:doc:`../fwinterface/fwflag_model` lr)
		* Extremely randomized trees (:doc:`../fwinterface/fwflag_model` etc)


Levels of analysis and group frequency threshold
------------------------------------------------

* See :doc:`../fwinterface/fwflag_group_freq_thresh`
* County level:

	* Ok to push the boundaries for p (use lots of features compared to observations), because you have well-estimated features
	* GFT: 20 to 50k range; (if really good data, use 50k)

* User-level:

	* Above rules for p directly apply. 
	* GFT: 1k (500 if N < 5k; 2k if N > 100k usually has absolutely no benefit)

* Message-level:

	* Rules above apply except, normally best to use binary encoding of 1to3grams 
	* GFT: 1 (but depends on task)
