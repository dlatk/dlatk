#!/usr/bin/bash 

while getopts "hd:t:c:f:-:" opt; do
    case $opt in
	h) echo "Usage - bash add_ngrams.sh -d <DB> -t <TABLE> -c <GROUP_FIELD> --group_freq_thresh <GFT> --set_p_occ <OCC> --set_pmi_threshold <PMI>" >&2
	   exit 2 ;;
        d) DATABASE=$OPTARG ;;
	t) TABLE=$OPTARG ;;
	c) GROUP_FIELD=$OPTARG ;;
	f) FEAT_TABLE=$OPTARG ;;
	-)
            case $OPTARG in
                group_freq_thresh) GFT="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                outcome_table) OT="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                outcomes) OC="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                feature_selection) FS="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
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

if [[ -v FS ]];
then
    echo "Classification with feature selection - " $FS
    dlatkInterface.py -d $DATABASE -t $TABLE -c $GROUP_FIELD --group_freq_thresh $GFT --feat_table $FEAT_TABLE --outcome_table $OT --outcomes $OC --nfold_test_classifiers --categories_to_binary $OC --feature_selection $FS --model lr --folds 10 --csv --output_name $OUTPUT
else
    echo "Classification without feature selection..."
    dlatkInterface.py -d $DATABASE -t $TABLE -c $GROUP_FIELD --group_freq_thresh $GFT --feat_table $FEAT_TABLE --outcome_table $OT --outcomes $OC --nfold_test_classifiers --categories_to_binary $OC --model lr --folds 10 --csv --output_name $OUTPUT
fi
