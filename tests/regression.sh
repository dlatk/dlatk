#!/usr/bin/env bash
# Cross-validated regression (default: 10-fold ridgecv), writing results to <OUTPUT>.accuracy_data.csv
source "$(dirname "$0")/common.sh"
USAGE="-d <DB> -t <TABLE> -c <GROUP_FIELD> -f '<FEAT_TABLE(S)>' --outcome_table <OUTCOME_TABLE> --outcomes '<OUTCOMES>' --output_name <OUTPUT> [-e <ENGINE>] [--group_freq_thresh <GFT>] [--feature_selection <FS_ALGORITHM>] [--model <MODEL> (default ridgecv)] [--folds <K> (default 10)]"

MODEL=ridgecv
FOLDS=10
while [ $# -gt 0 ]; do
    case $1 in
        --feature_selection) FS=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --folds) FOLDS=$2; shift 2 ;;
        *) parse_common "$@" || usage_exit; shift $SHIFT ;;
    esac
done
require DATABASE TABLE GROUP_FIELD FEAT_TABLES OT OUTCOMES OUTPUT

dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --group_freq_thresh $GFT \
    -f $FEAT_TABLES --outcome_table $OT --outcomes $OUTCOMES \
    --nfold_test_regression --model $MODEL --folds $FOLDS ${FS:+--feature_selection $FS} \
    --csv --output_name $OUTPUT
