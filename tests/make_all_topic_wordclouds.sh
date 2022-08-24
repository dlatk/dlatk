#!/usr/bin/bash 

#https://stackoverflow.com/questions/402377/using-getopts-to-process-long-and-short-command-line-options
while getopts "h-:" opt; do
    case $opt in
	h) echo "Usage - bash add_ngrams.sh -d <DB> -t <TABLE> -c <GROUP_FIELD> --group_freq_thresh <GFT> --set_p_occ <OCC> --set_pmi_threshold <PMI>" >&2
	   exit 2 ;;
	-)
            case $OPTARG in
		#https://www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html
                group_freq_thresh) GFT="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                topic_lexicon) TOPIC_LEX="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
                output) OUTPUT="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
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

dlatkInterface.py --topic_lexicon $TOPIC_LEX --group_freq_thresh $GFT --make_all_topic_wordclouds --tagcloud_colorscheme blue --output $OUTPUT
