#!/usr/bin/bash 

while getopts "hd:t:c:f:l:" opt; do
    case $opt in
	h) echo "Usage - bash unit_tests.sh -d <DB> -t <TABLE> -c <GROUP_FIELD> -l <LEX_TABLE> -o <OUTPUT>" >&2
	   exit 2 ;;
        d) DATABASE=$OPTARG ;;
	t) TABLE=$OPTARG ;;
	c) GROUP_FIELD=$OPTARG ;;
	f) FEAT_TABLE=$OPTARG ;;
	l) LEX_TABLE=$OPTARG ;;
	o) OUPUT=$OPTARG ;;
	-)
            case $OPTARG in
                group_freq_thresh) GFT="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                outcome_table) OT="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                outcomes) OC="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                controls) CTRLS="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                categories_to_binary) CTB="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                output_name) OUTPUT="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
		
		*)
		    if [ "$OPTERR" == 1 ]; then
                        echo "Non-option argument: '-${OPTARG}'" >&2
                    fi;;
            esac;;
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

#Test topic wordcloud creation
bash make_all_topic_wordclouds.sh --topic_lexicon $TABLE --group_freq_thresh 500 --output $OUTPUT

#Test feature correlation and wordcloud creation
bash correlate.sh -d $DATABASE -t $TABLE -c $GROUP_FIELD -f $FEAT_TABLE --group_freq_thresh $GFT --outcome_table $OT --outcomes $OC --controls $CTRLS --categories_to_binary $CTB --output_name $OUTPUT

#Regression
#dlatkInterface.py -d $1 -t $2 -c $3 --outcome_table $4 --outcomes age  --group_freq_thresh 500 --feat_table $5 --output_name $6 --nfold_test_regression --model ridgecv --folds 10 --csv
#dlatkInterface.py -d $1 -t $2 -c $3 --outcome_table $4 --outcomes age  --group_freq_thresh 500 --feat_table $5 --output_name $6 --nfold_test_regression --feature_selection pca --model ridgecv --folds 10 --csv
#dlatkInterface.py -d $1 -t $2 -c $3 --outcome_table $4 --outcomes age  --group_freq_thresh 500 --feat_table $5 --output_name $6 --nfold_test_regression --feature_selection magic_sauce --model ridgecv --folds 10 --csv

#Classification
#dlatkInterface.py -d $1 -t $2 -c $3 --outcome_table $4 --outcomes occu  --group_freq_thresh 500 --feat_table $5 --output_name $6 --nfold_test_classifiers --categories_to_binary occu --model lr --folds 10 --csv
#dlatkInterface.py -d $1 -t $2 -c $3 --outcome_table $4 --outcomes occu  --group_freq_thresh 500 --feat_table $5 --output_name $6 --nfold_test_classifiers --categories_to_binary occu --feature_selection pca --model lr --folds 10 --csv
#dlatkInterface.py -d $1 -t $2 -c $3 --outcome_table $4 --outcomes occu  --group_freq_thresh 500 --feat_table $5 --output_name $6 --nfold_test_classifiers --categories_to_binary occu --feature_selection magic_sauce --model lr --folds 10 --csv
