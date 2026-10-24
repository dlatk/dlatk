#!/usr/bin/env bash
# Message cleaning; each option writes a new message table:
#   --language_filter <LANG>  ->  <TABLE>_<LANG> (also --clean_messages; needs the langid module)
#   --deduplicate             ->  <TABLE>_dedup
#   --spam_filter             ->  <TABLE>_nospam
source "$(dirname "$0")/common.sh"
USAGE="-d <DB> -t <TABLE> -c <GROUP_FIELD> [-e <ENGINE>] [--language_filter <LANGUAGE>] [--deduplicate] [--spam_filter] (at least one)"

while [ $# -gt 0 ]; do
    case $1 in
        --language_filter) LF=$2; shift 2 ;;
        --deduplicate) DEDUP=1; shift ;;
        --spam_filter) SPAM=1; shift ;;
        *) parse_common "$@" || usage_exit; shift $SHIFT ;;
    esac
done
require DATABASE TABLE GROUP_FIELD
[ -n "$LF$DEDUP$SPAM" ] || usage_exit
set -e

[ -z "$LF" ] || dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --language_filter $LF --clean_messages
[ -z "$DEDUP" ] || dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --deduplicate
[ -z "$SPAM" ] || dlatk -e $ENGINE -d $DATABASE -t $TABLE -c $GROUP_FIELD --spam_filter
