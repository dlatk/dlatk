#!/usr/bin/bash 

while getopts "he:d:t:c:f:l:-:" opt; do
    case $opt in
	h) echo "Usage - bash unit_tests.sh -d <DB> -t <TABLE> -c <GROUP_FIELD> -f <FEAT_TABLE> -l <LEX_TABLE> --outcome_table <OUTCOME_TABLE> --freq_table <TOPIC_FREQ_TABLE> --classification_outcome '<CATEGORICAL_OUTCOMES>' --regression_outcome '<REAL_VALUED_OUTCOMES>' --output_folder <OUTPUT_FOLDER>" >&2
	   exit 2 ;;
        e) ENGINE=$OPTARG ;;
        d) DATABASE=$OPTARG ;;
	t) TABLE=$OPTARG ;;
	c) GROUP_FIELD=$OPTARG ;;
	f) FEAT_TABLE=$OPTARG ;;
	l) LEX_TABLE=$OPTARG ;;
	-)
            case $OPTARG in
                outcome_table) OT="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                freq_table) FT="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                classification_outcome) CO="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                regression_outcome) RO="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                output_folder) OUTPUT="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
		
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

#Create $OUTPUT folder
mkdir -p $OUTPUT

#Test message table cleaning.
bash clean_messages.sh -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --language_filter en

#Test ngram extraction, where n is upto 3.
bash add_ngrams.sh -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --group_freq_thresh 500 --set_p_occ 0.05 --set_pmi_threshold 3

#Test lexicon feature extraction
bash add_lex_table.sh -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD -l $LEX_TABLE --group_freq_thresh 500

#Test topic wordcloud creation
TOPIC_OUTPUT=$OUTPUT/topic_wordclouds
bash make_all_topic_wordclouds.sh -e $ENGINE --topic_lexicon $FT --group_freq_thresh 500 --output $TOPIC_OUTPUT

#Test feature correlation and wordcloud creation
CORREL_OUTPUT=$OUTPUT/correlations
mkdir -p $CORREL_OUTPUT
bash correlate.sh -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD -f $FEAT_TABLE --group_freq_thresh 500 --outcome_table $OT --outcomes $CO --controls $RO --categories_to_binary $CO --output_name $CORREL_OUTPUT/correlations

PREDICT_OUTPUT=$OUTPUT/predictions
mkdir -p $PREDICT_OUTPUT

#Test regression
bash regression.sh -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --outcome_table $OT --outcomes $RO  --group_freq_thresh 500 -f $FEAT_TABLE --output_name ${PREDICT_OUTPUT}/regression --feature_selection magic_sauce

#Test classification
bash classification.sh -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --outcome_table $OT --outcomes $CO  --group_freq_thresh 500 -f $FEAT_TABLE --output_name ${PREDICT_OUTPUT}/classification --feature_selection magic_sauce
