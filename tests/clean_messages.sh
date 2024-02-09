#!/usr/bin/bash 

while getopts "he:d:t:c:-:" opt; do
    case $opt in
	h) echo "Usage - bash clean_messages.sh -e <ENGINE> -d <DB> -t <TABLE> -c <GROUP_FIELD> --language_filter <LANGUAGE>" >&2
	   exit 2 ;;
        e) ENGINE=$OPTARG ;;
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

echo "TESTING --language_filter AND --clean_messages..."
dlatkInterface.py -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --language_filter $LF --clean_messages

echo "TESTING --deduplicate..."
dlatkInterface.py -e $ENGINE -d $DATABASE -t ${TABLE} -c $GROUP_FIELD --deduplicate

echo "TESTING --spam_filter..."
dlatkInterface.py -e $ENGINE -d $DATABASE -t ${TABLE} -c $GROUP_FIELD --spam_filter
