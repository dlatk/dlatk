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

dlatkInterface.py -d $DATABASE -t $TABLE -c $GROUP_FIELD --group_freq_thresh $GFT --correlate --tagcloud --make_wordclouds --rmatrix --csv --sort --feat_table $FEAT_TABLE --outcome_table $OT --outcomes $OC --controls $CTRLS --categories_to_binary $CTB --output_name $OUTPUT
