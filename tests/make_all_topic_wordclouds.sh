#!/usr/bin/env bash
# Wordcloud images for every topic in a topic lexicon, written to <OUTPUT>_alltopics/
source "$(dirname "$0")/common.sh"
USAGE="-d <DB> -t <TABLE> -c <GROUP_FIELD> --topic_lexicon <TOPIC_LEX> --output <OUTPUT> [-e <ENGINE>] [--lexicondb <LEXICON_DB> (default dlatk_lexica)]"

while [ $# -gt 0 ]; do
    case $1 in
        --topic_lexicon) TOPIC_LEX=$2; shift 2 ;;
        --lexicondb) LEXDB=$2; shift 2 ;;
        *) parse_common "$@" || usage_exit; shift $SHIFT ;;
    esac
done
require DATABASE TABLE GROUP_FIELD TOPIC_LEX OUTPUT

dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD ${LEXDB:+--lexicondb $LEXDB} \
    --topic_lexicon $TOPIC_LEX --make_all_topic_wordclouds --tagcloud_colorscheme blue --output $OUTPUT
