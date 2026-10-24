#!/usr/bin/env bash
# Lexicon (e.g. topic) feature extraction with a weighted lexicon.
source "$(dirname "$0")/common.sh"
USAGE="-d <DB> -t <TABLE> -c <GROUP_FIELD> -l <LEX_TABLE> [-e <ENGINE>] [--group_freq_thresh <GFT>]"

while [ $# -gt 0 ]; do
    case $1 in
        -l) LEX_TABLE=$2; shift 2 ;;
        *) parse_common "$@" || usage_exit; shift $SHIFT ;;
    esac
done
require DATABASE TABLE GROUP_FIELD LEX_TABLE

dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --group_freq_thresh $GFT \
    --add_lex_table -l $LEX_TABLE --weighted_lexicon
