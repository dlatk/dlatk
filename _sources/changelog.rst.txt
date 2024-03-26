.. _changelog:
=========
Changelog
=========

1.2.0 (2021-03-11)
------------------
  - New feature: Huggingface interface with --add_embedding (Bert, Roberta, XLNet, GPT2, etc.)
  - New feature: pymallet interface for creating LDA topics
  - New feature: optional sqlite backend
  - New feature: MySQL configuration read from config files (~/.my.cnf) instead of command line
  - Improving feature: dropped the 16to16 notation for default encoding (i.e., non-transformed) feature tables
  - Improving feature: new notation for group flag (--group or -g), in addition to previous version (--correl_field or -c)
  - Improving feature: default for running n-fold cross validation is to use all controls, use --all_control_combinations for all combinations
  - Improving feature: additional classification metrics printed to terminal
  - Improving feature: support for chinese segmentation in mallet
  - Improving feature: normalize lexicon scores by words with category (rather than all words), --lexicon_weighting
  - Improving feature: all external tools moved to ~/dlatk_tools
  - Bug: fixed bugs in --regression_to_lex

1.1.6 (2019-07-31)
------------------
  - New feature: bert feature extraction v1 ready
  - New feature: added --keep_low_variance_outcomes
  - New feature: added --outliers_to_mean to classifyPredictor
  - New feature: added --cohens_d flag
  - New feature: added factor adaptation code
  - New feature: added --multiclass flag
  - New feature: added new flag --predict_probabilities_to_feats
  - Improving feature: added effect size ranges to ngram wordcloud filenames
  - Improving feature: new default is to *not* print duplicate topic wordclouds, use --keep_duplicates to turn this off
  - Improving feature: beta classification auc ensembling
  - Improving feature: add _intercept=1 to all groups in lex feature tables
  - Improving feature: --classification_to_lexicon works with topic level features and multiple feature tables
  - Improving feature: remove underscores when uploading multiword lexica, use --keep_underscores to turn off
  - Improving feature: added mysql methods for checking indices
  - Improving feature: docker documentation
  - Improving feature: cleaned up regression predictor output
  - Improving feature: using default random seed throughout classes
  - Improving feature: fixed bug in interactions and language filtering
  - Improving feature: added printing of true values to pred_csv
  - Improving feature: teal colorscheme, aliases for 'tagcloud' and 'wordcloud'
  - Improving feature: metric names in wordcloud output
  - Bug: fixed n_iter shufflesplit outdated param
  - Bug: fixed nan pvalue issue in control corrections
  - Bug: fixed --log transform table naming convention
  - Bug: fixed bug in dlatk/classifyPredictor.py, scaler fit_transform was called during _multiXpredict, changed to transform
  - Bug: correct version of wordcloud python module is 1.1.3

1.1.5 (2018-03-02)
------------------
  - Improving feature: changes to LDA process: topicExtractor now accessible inside dlatkInterface
  - Improving feature: --lex_interface: lexInterface flags accessible inside dlatkInterface
  - Improving feature: changes default lexicon based feature table names
  - Improving feature: --categorical flag removes one outcome for any column with only two values.
  - Improving feature: mysqlMethods changed to mysqlmethods
  - Improving feature: default lexicon MySQL database now called "dlatk_lexica" (old database called "permaLexicon")
  - Bug fix: updated ridgehighcv to fix integer bug in sklearn

1.1.4 (2018-01-11)
------------------
  - Bug fix: added __init__.py file to Tools directory to fix pip install

1.1.3 (2018-01-09)
------------------
  - Improving feature: created the alias --show_feat_tables for --ls

1.1.2 (2018-01-09)
------------------
  - New Feature: added --show_tables
  - New Feature: added --describe_tables
  - New Feature: added --create_random_sample

1.1.1 (2017-11-30)
------------------
  - New Feature: added --categories_to_binary
  - New Feature: added --create_collocation_scores
  - New Feature: added --weighted_sample for running weight regression
  - New Feature: added --reduced_lexicon flag for creating super topics
  - New Feature: added --extension
  - Improving feature: allow LDA messages to have non-numeric message id
  - Improving feature: removed extra '.' from tagcloud filenames
  - Improving feature: moved location of Stanford Segmenter to dlaConstants
  - Bug fix: in --predict_regression_to_outcome_table and --predict_regression_to_feats
  - Bug fix: in setup.py for adding Tools directory
  - Bug fix: fixed bug in super topics (added auto increment id) 

1.1.0 (2017-06-29)
------------------
  - New feature: added --n_components / --num_factors flag to --fit_reducer
  - New feature: --clean_cloud flag working for n_gram correlation, all clouds, and topic clouds
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