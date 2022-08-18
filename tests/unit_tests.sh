#!/usr/bin/bash 

while getopts "hd:t:c:l:" opt; do
    case $opt in
	h) echo "Usage - bash unit_tests.sh -d <DB> -t <TABLE> -c <GROUP_FIELD> -l <LEX_TABLE>" ;;
        d) DATABASE=$OPTARG ;;
	t) TABLE=$OPTARG ;;
	c) GROUP_FIELD=$OPTARG ;;
	l) LEX_TABLE=$OPTARG ;;
	*) 
	    if [ "$OPTERR" == 1 ]; then
                echo "Non-option argument: '-${OPTARG}'" >&2
            fi;;
    esac
done

#Test ngram extraction, where n is upto 3.
bash add_ngrams.sh -d $DATABASE -t $TABLE -c $GROUP_FIELD --group_freq_thresh 500 --feat_occ_filter --set_p_occ 0.05 --feat_colloc_filter --set_pmi_threshold 3

#Test lexicon feature extraction
bash add_lex_table.sh -d $DATABASE -t $TABLE -c $GROUP_FIELD -l $LEX_TABLE --group_freq_thresh 500

#test imports
#dlatkInterface.py --help

#N-gram feature extraction where N=1,2,3
#dlatkInterface.py -d $1 -t $2 -c $3 --add_ngrams -n 1 2 3 --group_freq_thresh 500 --combine_feat_tables 1to3gram --feat_occ_filter --set_p_occ 0.05 --feat_colloc_filter --set_pmi_threshold 3

#LIWC2015 feature extraction
#dlatkInterface.py -d $1 -t $2 -c $3 --add_lex_table -l $4

#FB2k topic feature extraction
#dlatkInterface.py -d $1 -t $2 -c $3 --group_freq_thresh 500 --add_lex_table -l $4 --weighted_lexicon 

#Produce topic wordclouds
#dlatkInterface.py --topic_lexicon $1 --group_freq_thresh 500 --make_all_topic_wordclouds --tagcloud_colorscheme blue --output $2

#Language correlation with features
#dlatkInterface.py -d $1 -t $2 -c $3 --group_freq_thresh 500 --correlate --tagcloud --make_wordclouds --rmatrix --csv --sort --feat_table $4 --outcome_table $5 --outcomes age --controls is_female --output_name $6
#dlatkInterface.py -d $1 -t $2 -c $3 --group_freq_thresh 500 --correlate --tagcloud --make_wordclouds --rmatrix --csv --sort --feat_table $4 --outcome_table $5 --outcomes occu --controls age is_female --categories_to_binary occu --output_name $6

#Regression
#dlatkInterface.py -d $1 -t $2 -c $3 --outcome_table $4 --outcomes age  --group_freq_thresh 500 --feat_table $5 --output_name $6 --nfold_test_regression --model ridgecv --folds 10 --csv
#dlatkInterface.py -d $1 -t $2 -c $3 --outcome_table $4 --outcomes age  --group_freq_thresh 500 --feat_table $5 --output_name $6 --nfold_test_regression --feature_selection pca --model ridgecv --folds 10 --csv
#dlatkInterface.py -d $1 -t $2 -c $3 --outcome_table $4 --outcomes age  --group_freq_thresh 500 --feat_table $5 --output_name $6 --nfold_test_regression --feature_selection magic_sauce --model ridgecv --folds 10 --csv

#Classification
#dlatkInterface.py -d $1 -t $2 -c $3 --outcome_table $4 --outcomes occu  --group_freq_thresh 500 --feat_table $5 --output_name $6 --nfold_test_classifiers --categories_to_binary occu --model lr --folds 10 --csv
#dlatkInterface.py -d $1 -t $2 -c $3 --outcome_table $4 --outcomes occu  --group_freq_thresh 500 --feat_table $5 --output_name $6 --nfold_test_classifiers --categories_to_binary occu --feature_selection pca --model lr --folds 10 --csv
#dlatkInterface.py -d $1 -t $2 -c $3 --outcome_table $4 --outcomes occu  --group_freq_thresh 500 --feat_table $5 --output_name $6 --nfold_test_classifiers --categories_to_binary occu --feature_selection magic_sauce --model lr --folds 10 --csv
