.. _fwflag_classification_to_lexicon:
===========================
--classification_to_lexicon
===========================
Switch
======

--classification_to_lexicon

Description
===========

Extracts the coefficients from a classification model and turns them into a lexicon.

Argument and Default Value
==========================

Name of the lexicon to be created.

Details
=======

Use this switch to create a lexicon from a classification model. Either create the lexicon from a previously created model (using :doc:`fwflag_load_model`) or create a model using :doc:`fwflag_train_classifiers`. The name of the lexicon will be dd_ARGUMENT ; dd indicates the lexicon was data driven.

Only use classification algorithms that have linear coefficients (i.e by choosing the right :doc:`fwflag_model`), because the lexicon extraction equation won't make sense otherwise. This functionality hasn't totally been validated with advanced feature selection, so beware.
Also note that the coefficients won't be efficient to distinguish what features best characterize the outcomes looked at, use :doc:`fwflag_correlate` or other univariate techniques to get at that type of insight.


IMPORTANT

* This only works with binary classification, so if you want to do this with multiple classes (>2), create variables that are binary.
* When creating the model, use :doc:`fwflag_no_standardize` or the model will not make any sense.
* Multiple feature tables are allowed. 
* Lexica can be created with any *word level* feature (such as ngrams, dictionaries and topics) whose group norm is a relative frequency. Dictionaries and topics will be unrolled to the word level. 
* You cannot combined features with different group norm encodings (for example, binary 1grams and tf-idf 1grams). 

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes`
* :doc:`fwflag_no_standardize` 

Needs one of these two switches:

* :doc:`fwflag_train_classifiers`
* :doc:`fwflag_load_model` 

Example Commands
================


This command will train a logistic classification model to predict gender for users from 1grams (without standardizing) and create a lexicon called `dd_genderLex1grams`.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$16to16$0_01' --outcome_table blog_outcomes --outcomes gender --train_classifiers  --no_standardize --classification_to_lexicon genderLex1grams --model lr

This command will train a logistic classification model to predict age for users from 1grams and topics and create a lexicon called `dd_ageLex1gramsTopics`.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$16to16$0_1' 'feat$cat_met_a30_2000_cp_w$msgs$user_id$1gra' --outcome_table blog_outcomes --outcomes gender --train_classifiers --no_standardize --classification_to_lexicon ageLex1gramsTopics --model lr


References
==========

* Sap, M., Park, G., Eichstaedt, J., Kern, M., Stillwell, D., Kosinski, M., ... & Schwartz, H. A. (2014, October). Developing age and gender predictive lexica over social media. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 1146-1151).
