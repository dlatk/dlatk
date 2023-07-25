#!/usr/bin/bash 

while getopts "h-:" opt; do
    case $opt in
	h) echo "Usage - bash make_all_topic_wordclouds.sh --topic_lexicon <TOPIC_LEX> --group_freq_thresh <GFT> --output <OUTPUT>" >&2
	   exit 2 ;;
	-)
            case $OPTARG in
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

python ../dlatkInterface.py --topic_lexicon $TOPIC_LEX --group_freq_thresh $GFT --make_all_topic_wordclouds --tagcloud_colorscheme blue --output $OUTPUT
