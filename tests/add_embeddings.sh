#!/usr/bin/env bash
# Transformer embedding features (--add_emb_feat): message embeddings averaged per group.
# Writes feat$<EMB_TABLE_NAME>$<TABLE>$<GROUP_FIELD> when --emb_table_name is given.
source "$(dirname "$0")/common.sh"
USAGE="-d <DB> -t <TABLE> -c <GROUP_FIELD> [-e <ENGINE>] [--emb_model <MODEL> (default distilroberta-base)] [--emb_layers '<LAYERS>'] [--emb_table_name <NAME>]"

EMB_MODEL=distilroberta-base
while [ $# -gt 0 ]; do
    case $1 in
        --emb_model) EMB_MODEL=$2; shift 2 ;;
        --emb_layers) EMB_LAYERS=$2; shift 2 ;;
        --emb_table_name) EMB_NAME=$2; shift 2 ;;
        *) parse_common "$@" || usage_exit; shift $SHIFT ;;
    esac
done
require DATABASE TABLE GROUP_FIELD

dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --add_emb_feat --emb_model $EMB_MODEL \
    ${EMB_LAYERS:+--emb_layers $EMB_LAYERS} ${EMB_NAME:+--emb_table_name $EMB_NAME}
