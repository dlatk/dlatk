#!/usr/bin/env bash
# Smoothed copies of an n-gram feature table (--smooth_feats), one per method.
source "$(dirname "$0")/common.sh"
USAGE="-d <DB> -t <TABLE> -c <GROUP_FIELD> -f <NGRAM_FEAT_TABLE> [-e <ENGINE>] [--group_freq_thresh <GFT>] [--methods '<METHOD...>' (default 'dirichlet nb pln zinb logodds_idp logodds hurdle')] [--smooth_mu <MU>] [--smooth_log]"

METHODS="dirichlet nb pln zinb logodds_idp logodds hurdle"
while [ $# -gt 0 ]; do
    case $1 in
        --methods) METHODS=$2; shift 2 ;;
        --smooth_mu) MU=$2; shift 2 ;;
        --smooth_log) LOG=1; shift ;;
        *) parse_common "$@" || usage_exit; shift $SHIFT ;;
    esac
done
require DATABASE TABLE GROUP_FIELD FEAT_TABLES
set -e

for method in $METHODS; do
    dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD -f $FEAT_TABLES --group_freq_thresh $GFT \
        --smooth_feats $method ${MU:+--smooth_mu $MU} ${LOG:+--smooth_log}
done
