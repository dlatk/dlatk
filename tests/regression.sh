#!/usr/bin/bash 

while getopts "he:d:t:c:f:-:" opt; do
    case $opt in
	h) echo "Usage - bash regression.sh -d <DB> -t <TABLE> -c <GROUP_FIELD> -f <FEAT_TABLE> --group_freq_thresh <GFT> --outcome_table <OUTCOME_TABLE> --outcomes <OUTCOME> --feature_selection <FS_ALGORITHM> --output_name <OUTPUT>" >&2
	   exit 2 ;;
        e) ENGINE=$OPTARG ;;
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
    echo "Regression with feature selection - ", $FS
    python ../dlatkInterface.py -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --group_freq_thresh $GFT --feat_table $FEAT_TABLE --outcome_table $OT --outcomes $OC --nfold_test_regression --feature_selection $FS --model ridgecv --folds 10 --csv --output_name $OUTPUT
else
    echo "Regression without feature selection..."
    python ../dlatkInterface.py -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --group_freq_thresh $GFT --feat_table $FEAT_TABLE --outcome_table $OT --outcomes $OC --nfold_test_regression --model ridgecv --folds 10 --csv --output_name $OUTPUT
fi
