#!/usr/bin/bash 

#https://stackoverflow.com/questions/402377/using-getopts-to-process-long-and-short-command-line-options
while getopts "hd:t:c:-:" opt; do
    case $opt in
	h) echo "Usage - bash add_ngrams.sh -d <DB> -t <TABLE> -c <GROUP_FIELD> --group_freq_thresh <GFT> --set_p_occ <OCC> --set_pmi_threshold <PMI>" ;;
        d) DATABASE=$OPTARG ;;
	t) TABLE=$OPTARG ;;
	c) GROUP_FIELD=$OPTARG ;;
	-)
            case $OPTARG in
		#https://www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html
                group_freq_thresh) GFT="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                set_p_occ) OCC="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                set_pmi_threshold) PMI="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
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

#Call dlatkInterface.py with appropriate flags and values.
dlatkInterface.py -d $DATABASE -t $TABLE -c $GROUP_FIELD --add_ngrams -n 1 2 3 --group_freq_thresh $GFT --combine_feat_tables 1to3gram --feat_occ_filter --set_p_occ $OCC --feat_colloc_filter --set_pmi_threshold $PMI
