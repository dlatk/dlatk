#!/usr/bin/bash 

while getopts "hd:t:c:l:-:" opt; do
    case $opt in
	h) echo "Usage - bash add_ngrams.sh -d <DB> -t <TABLE> -c <GROUP_FIELD> --group_freq_thresh <GFT> --set_p_occ <OCC> --set_pmi_threshold <PMI>" ;;
        d) DATABASE=$OPTARG ;;
	t) TABLE=$OPTARG ;;
	c) GROUP_FIELD=$OPTARG ;;
	l) LEX_TABLE=$OPTARG ;;
	-)
            case $OPTARG in
                group_freq_thresh) GFT="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
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

dlatkInterface.py -d $DATABASE -t $TABLE -c $GROUP_FIELD --group_freq_thresh $GFT --add_lex_table -l $LEX_TABLE --weighted_lexicon 

