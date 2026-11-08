#!/usr/bin/env bash
# Keeps the features of an n-gram table that are reliable across groups (--feat_reliability_filter).
source "$(dirname "$0")/common.sh"
USAGE="-d <DB> -t <TABLE> -c <GROUP_FIELD> -f <NGRAM_FEAT_TABLE> [-e <ENGINE>] [--group_freq_thresh <GFT>] [--set_reliability <R>]"

while [ $# -gt 0 ]; do
    case $1 in
        --set_reliability) REL=$2; shift 2 ;;
        *) parse_common "$@" || usage_exit; shift $SHIFT ;;
    esac
done
require DATABASE TABLE GROUP_FIELD FEAT_TABLES
set -e

dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD -f $FEAT_TABLES --group_freq_thresh $GFT \
    --feat_reliability_filter ${REL:+--set_reliability $REL}
