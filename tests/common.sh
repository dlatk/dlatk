# Shared setup for the tests/*.sh helper scripts (sourced, not run).
# Runs the repo's own dlatkInterface.py, found relative to this tests/ folder.

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$TESTS_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Python < 3.9 block-buffers stderr when redirected, which hides progress in logs
export PYTHONUNBUFFERED=1

ENGINE=mysql
GFT=500
FEAT_TABLES=""

# dlatk ARGS...: prints and runs dlatkInterface.py from the repo root
dlatk() {
    echo "+ dlatkInterface.py $*"
    (cd "$ROOT" && "$PYTHON_BIN" "$ROOT/dlatkInterface.py" "$@")
}

usage_exit() {
    echo "Usage - bash $(basename "$0") $USAGE" >&2
    exit 2
}

# require VAR...: exits with usage if any of the named variables is empty
require() {
    for v in "$@"; do
        [ -n "${!v}" ] || { echo "missing required option for $v" >&2; usage_exit; }
    done
}

# parse_common ARGS...: parses one option shared by the helpers, setting SHIFT to the
# number of arguments it consumed; returns 1 for an option it does not know.
# -f may be repeated or given a space-separated list.
parse_common() {
    SHIFT=2
    case $1 in
        -h|--help) usage_exit ;;
        -e) ENGINE=$2 ;;
        -d) DATABASE=$2 ;;
        -t) TABLE=$2 ;;
        -c|-g) GROUP_FIELD=$2 ;;
        -f|--feat_table) FEAT_TABLES="$FEAT_TABLES $2" ;;
        --group_freq_thresh) GFT=$2 ;;
        --outcome_table) OT=$2 ;;
        --outcomes) OUTCOMES=$2 ;;
        --output_name|--output) OUTPUT=$2 ;;
        *) echo "unknown option: $1" >&2; return 1 ;;
    esac
}
