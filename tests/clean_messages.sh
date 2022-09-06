#!/usr/bin/bash 

while getopts "hd:t:c:-:" opt; do
    case $opt in
	h) echo "Usage - bash clean_messages.sh -d <DB> -t <TABLE> -c <GROUP_FIELD> --language_filter <LANGUAGE>" >&2
	   exit 2 ;;
        d) DATABASE=$OPTARG ;;
	t) TABLE=$OPTARG ;;
	c) GROUP_FIELD=$OPTARG ;;
	-)
            case $OPTARG in
                language_filter) LF="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
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

dlatkInterface.py -d $DATABASE -t $TABLE -c $GROUP_FIELD --language_filter $LF --clean_messages
dlatkInterface.py -d $DATABASE -t ${TABLE}_$LF -c $GROUP_FIELD --deduplicate --clean_messages
