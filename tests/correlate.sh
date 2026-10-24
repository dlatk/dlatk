#!/usr/bin/env bash
# DLA: correlates features with outcomes, writing an rmatrix, csv and tagclouds
# (topic tagclouds when --topic_lexicon is given). Lists are space-separated and quoted.
source "$(dirname "$0")/common.sh"
USAGE="-d <DB> -t <TABLE> -c <GROUP_FIELD> -f '<FEAT_TABLE(S)>' --outcome_table <OUTCOME_TABLE> --outcomes '<OUTCOMES>' --output_name <OUTPUT> [-e <ENGINE>] [--group_freq_thresh <GFT>] [--controls '<CONTROLS>'] [--categories_to_binary '<VARIABLES>'] [--topic_lexicon <TOPIC_LEX>] [--make_wordclouds]"

while [ $# -gt 0 ]; do
    case $1 in
        --controls) CTRLS=$2; shift 2 ;;
        --categories_to_binary) CTB=$2; shift 2 ;;
        --topic_lexicon) TOPIC_LEX=$2; shift 2 ;;
        --make_wordclouds) WORDCLOUDS=1; shift ;;
        *) parse_common "$@" || usage_exit; shift $SHIFT ;;
    esac
done
require DATABASE TABLE GROUP_FIELD FEAT_TABLES OT OUTCOMES OUTPUT

if [ -n "$TOPIC_LEX" ]; then
    CLOUDS="--topic_tagcloud --topic_lexicon $TOPIC_LEX ${WORDCLOUDS:+--make_topic_wordclouds}"
else
    CLOUDS="--tagcloud ${WORDCLOUDS:+--make_wordclouds}"
fi
dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --group_freq_thresh $GFT \
    -f $FEAT_TABLES --outcome_table $OT --outcomes $OUTCOMES \
    ${CTRLS:+--controls $CTRLS} ${CTB:+--categories_to_binary $CTB} \
    --correlate --rmatrix --csv --sort $CLOUDS --output_name $OUTPUT
