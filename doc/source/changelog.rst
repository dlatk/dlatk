.. _changelog:
=========
Changelog
=========

1.1.0 (2017-06-29)
------------------
  - New feature: added --n_components / --num_factors flag to --fit_reducer
  - New feature: --clean_cloud flag working for n_gram correlatin, all clouds, and topic clouds
  - New feature: added p correction to correlate --auc
  - New feature: messageAnnotator and messageTransformer classes added
  - New feature: clustering.py changed to dimensionReducer.py
  - New feature: FeatureWorker changed to DLAWorker
  - New feature: --clean_messages flag for anonymizing message tables
  - Improving feature: residualized control model in --combo_test_regression 
  - Improving feature: changed f1 to macro f1 instead of pos-class f1
  - Bug fix: in --predict_regression_to_outcome_table
  - Bug fix: in --regression_to_lexicon

1.0.1 (2016-11-21)
------------------
  - New feature: --deduplicate flag added to remove duplicate tweets within user
  - New feature: --spam_filter added
  - New feature: --cleanmessages added to --deduplicate and --spam_filter
  - New feature: --fold_column added to classifyPredictor and regressionPredictor
  - Bug fix: --print_csv
  - Bug fix: csv from binary to text mode lexInterface
  - Bug fix: add_postimexdiff, nlp server
  - Improving feature: error messages when incorrectly specifying --p_correction
  - Improving feature: changed _tok message tables to always be stored as longtext

1.0.0 (2016-10-21)
------------------
  - New feature: python 3.5 version, changed from FeatureWorker to dlatk
  - New feature: --densify_table
  - New feature: --language_filter and --clean_messages
  - New feature: --make_all_topic_wordclouds
  - New feature: --no_lower flag added for --add_char_ngrams
  - New feature: confidence intervals added to --correlate, --rmatrix and --csv
  - New feature: --add_message_id flag replaces LexicaInterface/addMessageID.py
  - New feature: --ls
  - New feature: stratified classification
  - New feature: --where
  - New feature: --stratify_folds added to --combo_test_classifiers
  - Bug fix: using multiple feature tables with --to_file
  - Improving feature: batch insert in tfidf
  - Improving feature: make_wordclouds/make_topic_wordcloud error catch
  - Improving feature: removed lda.py

0.6.1 (2016-07-05)
------------------
  - New feature: install notes for OSX
  - Bug fix: using multiple feature tables with --to_file
  - Improving feature: Default MySQL host changed from 'localhost' to 127.0.0.1

0.6.0 (2016-06-15)
------------------
  - Bug fix: in makeBlackWhiteList
  - Bug fix: in fiftyChecks, added random seed
  - Bug fix: for using old pickle files
  - Improving feature: added unicode try/except in wordcloud module and print csv
  - Improving feature: removed hardcoded utf8 and hardcoded table charset
  - New feature: --feat_selection_string
  - New feature: --add_corp_lex_table 
  - New feature: new models added to regressionPredictor (ridgefirstpasscv, ridgehighcv, ridgelowcv)
  - New feature: --add_char_ngrams
  - New feature: --no_outcomes and --no_controls 

0.5.0 (2016-04-01)
------------------


0.4.0 (2015-08-17)
------------------