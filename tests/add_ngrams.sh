#!/usr/bin/env bash
# N-gram extraction, combining the n-gram tables, and optional occurrence / PMI filters.
# Each filter is applied to the combined (unfiltered) table.
source "$(dirname "$0")/common.sh"
USAGE="-d <DB> -t <TABLE> -c <GROUP_FIELD> [-e <ENGINE>] [-n '<N...>' (default '1 2 3')] [--combine_feat_tables <NAME> (default 1to<max n>gram)] [--group_freq_thresh <GFT>] [--set_p_occ <OCC>] [--set_pmi_threshold <PMI>] [--no_extract (only run the filters)]"

NS="1 2 3"
EXTRACT=1
while [ $# -gt 0 ]; do
    case $1 in
        -n) NS=$2; shift 2 ;;
        --combine_feat_tables) COMBINE=$2; shift 2 ;;
        --set_p_occ) OCC=$2; shift 2 ;;
        --set_pmi_threshold) PMI=$2; shift 2 ;;
        --no_extract) EXTRACT=0; shift ;;
        *) parse_common "$@" || usage_exit; shift $SHIFT ;;
    esac
done
require DATABASE TABLE GROUP_FIELD
set -e

MAX_N=$(echo $NS | tr ' ' '\n' | sort -n | tail -1)
if [ "$(echo $NS | wc -w)" -gt 1 ]; then
    COMBINE=${COMBINE:-1to${MAX_N}gram}
    NGRAM_FEAT="feat\$${COMBINE}\$${TABLE}\$${GROUP_FIELD}"
else
    NGRAM_FEAT="feat\$${NS}gram\$${TABLE}\$${GROUP_FIELD}"
fi

if [ $EXTRACT -eq 1 ]; then
    dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --add_ngrams -n $NS \
        ${COMBINE:+--combine_feat_tables $COMBINE}
fi
if [ -n "$OCC" ]; then
    dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD -f "$NGRAM_FEAT" \
        --feat_occ_filter --set_p_occ $OCC --group_freq_thresh $GFT
fi
if [ -n "$PMI" ]; then
    dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD -f "$NGRAM_FEAT" \
        --feat_colloc_filter --set_pmi_threshold $PMI
fi
