.. _dlatkinterface_ordered:
****************************
dlatkInterface Flags by type
****************************

:doc:`fwinterface`

Setup
=====
* :doc:`fwinterface/fwflag_d`, --corpdb
* :doc:`fwinterface/fwflag_t`, --corptable
* :doc:`fwinterface/fwflag_c`, --correl_field
* :doc:`fwinterface/fwflag_h`, --host
* :doc:`fwinterface/fwflag_l`, --lex_table
* :doc:`fwinterface/fwflag_f`, --feat_table
* :doc:`fwinterface/fwflag_lexicondb` 
* :doc:`fwinterface/fwflag_outcome_table`
* :doc:`fwinterface/fwflag_word_table`
* :doc:`fwinterface/fwflag_group_freq_thresh`
* :doc:`fwinterface/fwflag_output_name`	
* :doc:`fwinterface/fwflag_messageid_field`
* :doc:`fwinterface/fwflag_message_field`
* :doc:`fwinterface/fwflag_date_field`
* :doc:`fwinterface/fwflag_to_file` 
* :doc:`fwinterface/fwflag_from_file`
* :doc:`fwinterface/fwflag_encoding`
* :doc:`fwinterface/fwflag_no_unicode`
* :doc:`fwinterface/fwflag_where`
* :doc:`fwinterface/fwflag_ls`
* :doc:`fwinterface/fwflag_colabify`

Preprocessing
=============
* :doc:`fwinterface/fwflag_add_tokenized` 
* :doc:`fwinterface/fwflag_add_parses` 
* :doc:`fwinterface/fwflag_add_segmented` 
* :doc:`fwinterface/fwflag_print_tokenized_lines`
* :doc:`fwinterface/fwflag_print_joined_feature_lines`
* :doc:`fwinterface/fwflag_add_tweetpos`
* :doc:`fwinterface/fwflag_add_tweettok`

Feature Extraction
==================
* :doc:`fwinterface/fwflag_add_lex_table` [-l LEX] 
* :doc:`fwinterface/fwflag_add_corp_lex_table` [-l LEX] 
* :doc:`fwinterface/fwflag_add_ngrams` [-n N [N2 …] ]
* :doc:`fwinterface/fwflag_add_ngrams_from_tokenized` [-n N [N2 …] ] 
* :doc:`fwinterface/fwflag_add_pos_ngram_table`
* :doc:`fwinterface/fwflag_add_pos_table`
* :doc:`fwinterface/fwflag_add_char_ngrams` [-n N [N2 …] ]
* :doc:`fwinterface/fwflag_anscombe`
* :doc:`fwinterface/fwflag_boolean`
* :doc:`fwinterface/fwflag_log`
* :doc:`fwinterface/fwflag_sqrt`
* :doc:`fwinterface/fwflag_use_collocs` 
* :doc:`fwinterface/fwflag_combine_feat_tables`
* :doc:`fwinterface/fwflag_lex_anscombe`
* :doc:`fwinterface/fwflag_lex_boolean`
* :doc:`fwinterface/fwflag_lex_sqrt`
* :doc:`fwinterface/fwflag_ex_log`

Feature Refinement
==================
* :doc:`fwinterface/fwflag_feat_names`
* :doc:`fwinterface/fwflag_whitelist`
* :doc:`fwinterface/fwflag_feat_whitelist`
* :doc:`fwinterface/fwflag_blacklist`
* :doc:`fwinterface/fwflag_feat_blacklist`
* :doc:`fwinterface/fwflag_add_lda_messages` 
* :doc:`fwinterface/fwflag_feat_correl_filter` 
* :doc:`fwinterface/fwflag_feat_colloc_filter` 
* :doc:`fwinterface/fwflag_feat_occ_filter` 
* :doc:`fwinterface/fwflag_feat_group_by_outcomes` 
* :doc:`fwinterface/fwflag_aggregate_feats_by_new_group` 
* :doc:`fwinterface/fwflag_p_value`
* :doc:`fwinterface/fwflag_tf_idf`

Language Insights
=================
* :doc:`fwinterface/fwflag_correlate`
* :doc:`fwinterface/fwflag_outcome_controls`
* :doc:`fwinterface/fwflag_interaction_ddla` 
* :doc:`fwinterface/fwflag_logistic_reg`
* :doc:`fwinterface/fwflag_mediation` 
* :doc:`fwinterface/fwflag_ttest_feat_tables` 
* :doc:`fwinterface/fwflag_rmatrix` 
* :doc:`fwinterface/fwflag_csv` 
* :doc:`fwinterface/fwflag_sort` 
* :doc:`fwinterface/fwflag_zScoreGroup`
* :doc:`fwinterface/fwflag_outcome_with_outcome`
* :doc:`fwinterface/fwflag_outcome_with_outcome_only`

Clustering
==========
* :doc:`fwinterface/fwflag_fit_reducer` 
* :doc:`fwinterface/fwflag_cca`
* :doc:`fwinterface/fwflag_cca_predict_components`

Prediction
==========
* :doc:`fwinterface/fwflag_sparse`
* :doc:`fwinterface/fwflag_prediction_csv`
* :doc:`fwinterface/fwflag_weighted_eval`
* :doc:`fwinterface/fwflag_folds`
* :doc:`fwinterface/fwflag_feature_selection`
* :doc:`fwinterface/fwflag_feature_selection_string`

Regression
----------
* :doc:`fwinterface/fwflag_combo_test_regression`
* :doc:`fwinterface/fwflag_train_regression`
* :doc:`fwinterface/fwflag_test_regression`
* :doc:`fwinterface/fwflag_predict_regression`
* :doc:`fwinterface/fwflag_predict_regression_to_feats`
* :doc:`fwinterface/fwflag_predict_regression_to_outcome_table`
* :doc:`fwinterface/fwflag_regression_to_lexicon`
* :doc:`fwinterface/fwflag_control_adjust_reg`

Classification
--------------
* :doc:`fwinterface/fwflag_combo_test_classifiers`
* :doc:`fwinterface/fwflag_train_classifiers`
* :doc:`fwinterface/fwflag_test_classifiers`
* :doc:`fwinterface/fwflag_predict_classifiers`
* :doc:`fwinterface/fwflag_predict_classifiers_to_feats`
* :doc:`fwinterface/fwflag_predict_classification_to_outcome_table`
* :doc:`fwinterface/fwflag_classification_to_lexicon`

Visualization
=============
* :doc:`fwinterface/fwflag_make_wordclouds`
* :doc:`fwinterface/fwflag_make_topic_wordclouds`
* :doc:`fwinterface/fwflag_tagcloud`
* :doc:`fwinterface/fwflag_topic_tagcloud`
* :doc:`fwinterface/fwflag_DDLATagcloud`
* :doc:`fwinterface/fwflag_tagcloud_colorscheme`
* :doc:`fwinterface/fwflag_max_tagcloud_words`
