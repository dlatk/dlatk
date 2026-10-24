#!/usr/bin/env bash
# Unit tests for dlatkInterface.py, covering the "Suggested Unit Tests" in
# dev_instructions.txt (following the DLA / prediction tutorials), plus extended tests.
#
# Uses only data distributed with dlatk (dla_tutorial.sql, dlatk_lexica.sql) and runs the
# repo's own dlatkInterface.py on a scratch sample of the message table, so `msgs` is untouched.
#
# Usage: bash tests/unit_tests.sh [-d DB] [-t SOURCE_MSG_TABLE] [-o OUTPUT_DIR]
#                                 [-n NUM_USERS] [-m MAX_MSGS] [-p P_OCC] [-c] [-k] [-s]
#   -n  number of users sampled (fixed seed) into the scratch table (default 150)
#   -m  maximum messages kept per user, 0 for no cap (default 30)
#   -p  occurrence filter threshold for the 1-2gram table used in DLA/prediction (default 0.2)
#   -c  core tests only (skip the extended tests)
#   -k  keep the scratch message table and its feature tables afterwards
#   -s  skip sampling and 1-2gram extraction: start at the occurrence filter, reusing the
#       scratch table and 1to2gram table kept by an earlier run with -k
#
# Requires: dla_tutorial (msgs, blog_outcomes) and dlatk_lexica (met_a30_2000_*).
# Extended tests also use Java (wordclouds), langid (language filter), and
# transformers / sentence-transformers (downloads distilroberta-base on first run); the language filter and sentence-transformer tests are
# reported as skipped when their dependencies are missing.

DB=dla_tutorial
SRC_TABLE=msgs
OUT=$(mktemp -d -t dlatk_unit_tests.XXXX)
KEEP=0
NUM_USERS=150
MAX_MSGS=30
P_OCC=0.2
SKIP_EXTRACT=0
CORE_ONLY=0
while getopts "hd:t:o:n:m:p:cks" opt; do
    case $opt in
        h) sed -n '2,21p' "$0"; exit 2 ;;
        d) DB=$OPTARG ;;
        t) SRC_TABLE=$OPTARG ;;
        o) OUT=$OPTARG ;;
        n) NUM_USERS=$OPTARG ;;
        m) MAX_MSGS=$OPTARG ;;
        p) P_OCC=$OPTARG ;;
        c) CORE_ONLY=1 ;;
        k) KEEP=1 ;;
        s) SKIP_EXTRACT=1 ;;
        *) exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Python < 3.9 block-buffers stderr when redirected, which hides progress in the logs
export PYTHONUNBUFFERED=1
DLATK=("$PYTHON_BIN" "$ROOT/dlatkInterface.py")
T=${SRC_TABLE}_unittest
G=user_id
GFT=500
OT=blog_outcomes
TOPIC_CP=met_a30_2000_cp
TOPIC_FREQ=met_a30_2000_freq_t50ll
NGRAM_FEAT="feat\$1to2gram\$${T}\$${G}"
NGRAM_FEAT_OCC="${NGRAM_FEAT}\$${P_OCC//./_}"
NGRAM_FEAT_PMI="${NGRAM_FEAT}\$pmi3_0"
TOPIC_FEAT="feat\$cat_${TOPIC_CP}_w\$${T}\$${G}\$1gra"
SMALL_TOPIC_LEX=${T}_topics
EMB_MODEL=distilroberta-base
EMB_T=${T}_emb
EMB_FEAT="feat\$distilroberta\$${EMB_T}\$${G}"

mkdir -p "$OUT"
PASSED=(); FAILED=(); SKIPPED=()

# run NAME CMD...: runs a dlatk command, logging to $OUT/NAME.log
run() {
    local name=$1; shift
    echo "=== $name"
    (cd "$ROOT" && "$@") > "$OUT/$name.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] || echo "    command failed (exit $rc); see $OUT/$name.log"
    return $rc
}

# check NAME CONDITION_DESC TEST...: records pass/fail for a step
check() {
    local name=$1 desc=$2; shift 2
    if "$@"; then
        PASSED+=("$name")
        echo "    PASS"
    else
        FAILED+=("$name")
        echo "    FAIL: $desc"
    fi
}

skip() {
    echo "=== $1"
    echo "    SKIP: $2"
    SKIPPED+=("$1")
}

table_exists() {
    [ -n "$(mysql -Ne "SHOW TABLES FROM $DB LIKE '$1'" 2>/dev/null)" ]
}

table_nonempty() {
    table_exists "$1" && [ "$(mysql -Ne "SELECT COUNT(*) FROM $DB.\`$1\`")" -gt 0 ]
}

num_feats() {
    mysql -Ne "SELECT COUNT(DISTINCT feat) FROM $DB.\`$1\`" 2>/dev/null
}

files_exist() {
    for f in "$@"; do [ -s "$f" ] || { echo "    missing: $f"; return 1; }; done
}

dir_nonempty() {
    [ -n "$(ls -A "$1" 2>/dev/null)" ] || { echo "    missing or empty: $1"; return 1; }
}

# cap_messages TABLE MAX: keeps each group's earliest MAX messages (by message_id).
# Uses a self-join rather than window functions so it works on MySQL 5.x as well as 8.x / MariaDB.
cap_messages() {
    mysql "$DB" -e "DROP TABLE IF EXISTS ${1}_keep;
        CREATE TABLE ${1}_keep (message_id BIGINT PRIMARY KEY)
            SELECT a.message_id FROM $1 a JOIN $1 b ON a.$G = b.$G AND b.message_id <= a.message_id
            GROUP BY a.message_id HAVING COUNT(*) <= $2;
        DELETE FROM $1 WHERE message_id NOT IN (SELECT message_id FROM ${1}_keep);
        DROP TABLE ${1}_keep;"
}

# Drops the scratch table, tables derived from it (msgs_unittest_en, _dedup, ...) and their feature tables
drop_scratch() {
    mysql -Ne "SHOW TABLES FROM $DB" | grep -E "^${T}(_[a-z0-9]+)?\$|\\\$${T}(_[a-z0-9]+)?\\\$" | while read -r tbl; do
        mysql -e "DROP TABLE IF EXISTS $DB.\`$tbl\`"
    done
}

echo "Output folder: $OUT"
echo "Scratch message table: $DB.$T"

##################
## CORE TESTS

if [ "$SKIP_EXTRACT" -eq 1 ]; then
    # Reuse the scratch table and 1to2gram table; drop only what later steps rebuild
    table_nonempty "$T" && table_nonempty "$NGRAM_FEAT" \
        || { echo "-s needs $DB.$T and $NGRAM_FEAT from an earlier run with -k"; exit 1; }
    for tbl in "$NGRAM_FEAT_OCC" "$NGRAM_FEAT_PMI" "$TOPIC_FEAT"; do
        mysql -e "DROP TABLE IF EXISTS $DB.\`$tbl\`"
    done
    echo "Skipping extraction; reusing $DB.$T and $NGRAM_FEAT"
else
    # Fresh scratch table with a fixed random sample of users
    # (and drop tables left by earlier runs)
    drop_scratch
    mysql -e "CREATE TABLE $DB.$T LIKE $DB.$SRC_TABLE;
        INSERT INTO $DB.$T SELECT m.* FROM $DB.$SRC_TABLE m
        JOIN (SELECT DISTINCT $G FROM $DB.$SRC_TABLE ORDER BY RAND(42) LIMIT $NUM_USERS) u USING ($G)" \
        || { echo "Could not create scratch table $DB.$T"; exit 1; }

    if [ "$MAX_MSGS" -gt 0 ]; then
        cap_messages "$T" "$MAX_MSGS" || { echo "Could not cap messages per user in $DB.$T"; exit 1; }
    fi
    echo "Scratch table: $(mysql -Ne "SELECT CONCAT(COUNT(*), ' messages, ', COUNT(DISTINCT $G), ' users') FROM $DB.$T")"

    # N-gram extraction (1-2 grams) and combined feature table
    run ngrams "${DLATK[@]}" -d $DB -t $T -c $G \
        --add_ngrams -n 1 2 --combine_feat_tables 1to2gram
    check ngrams "1to2gram feature table populated" table_nonempty "$NGRAM_FEAT"
fi

# Occurrence filter: keep features used by at least P_OCC of groups
run ngrams_occ "${DLATK[@]}" -d $DB -t $T -c $G -f "$NGRAM_FEAT" \
    --feat_occ_filter --set_p_occ $P_OCC --group_freq_thresh $GFT
check ngrams_occ "p_occ-filtered feature table populated" table_nonempty "$NGRAM_FEAT_OCC"

# Topic extraction (weighted lexicon of Facebook topics)
run topics "${DLATK[@]}" -d $DB -t $T -c $G \
    --add_lex_table -l $TOPIC_CP --weighted_lexicon
check topics "weighted topic feature table populated" table_nonempty "$TOPIC_FEAT"

# Word DLA: rmatrix, csv, tagclouds; age + occupation (categories to binary)
run dla_ngrams "${DLATK[@]}" -d $DB -t $T -c $G -f "$NGRAM_FEAT_OCC" \
    --outcome_table $OT --group_freq_thresh $GFT \
    --outcomes age occu --categories_to_binary occu \
    --correlate --rmatrix --csv --sort --tagcloud \
    --output_name "$OUT/dla_ngrams"
check dla_ngrams "rmatrix html/csv and tagcloud output written" \
    files_exist "$OUT/dla_ngrams.html" "$OUT/dla_ngrams.csv" "$OUT/dla_ngrams_tagcloud.txt"

# Topic DLA with topic tagclouds
run dla_topics "${DLATK[@]}" -d $DB -t $T -c $G -f "$TOPIC_FEAT" \
    --outcome_table $OT --group_freq_thresh $GFT \
    --outcomes age occu --categories_to_binary occu \
    --correlate --rmatrix --csv --sort \
    --topic_tagcloud --topic_lexicon $TOPIC_FREQ \
    --output_name "$OUT/dla_topics"
check dla_topics "rmatrix csv and topic tagcloud output written" \
    files_exist "$OUT/dla_topics.csv" "$OUT/dla_topics_topic_tagcloud.txt"

# Ridge regression, 10-fold cross-validation
run regression "${DLATK[@]}" -d $DB -t $T -c $G -f "$TOPIC_FEAT" "$NGRAM_FEAT_OCC" \
    --outcome_table $OT --group_freq_thresh $GFT --outcomes age \
    --nfold_test_regression --model ridgecv --folds 10 --csv \
    --output_name "$OUT/regression"
check regression "regression csv written" files_exist "$OUT/regression.accuracy_data.csv"

# Logistic regression classification, 10-fold, on an occupation indicator.
# (--categories_to_binary occu makes classes too rare for 10 folds on a small sample;
# the DLA steps above cover categories_to_binary.)
run classification "${DLATK[@]}" -d $DB -t $T -c $G -f "$TOPIC_FEAT" "$NGRAM_FEAT_OCC" \
    --outcome_table $OT --group_freq_thresh $GFT \
    --outcomes is_student \
    --nfold_test_classifiers --model lr --folds 10 --csv \
    --output_name "$OUT/classification"
check classification "classification csv written" files_exist "$OUT/classification.accuracy_data.csv"

##################
## EXTENDED TESTS

if [ "$CORE_ONLY" -eq 0 ]; then

    # PMI collocation filter
    run ngrams_pmi "${DLATK[@]}" -d $DB -t $T -c $G -f "$NGRAM_FEAT" \
        --feat_colloc_filter --set_pmi_threshold 3
    check ngrams_pmi "pmi-filtered feature table populated" table_nonempty "$NGRAM_FEAT_PMI"

    # Word DLA with a control variable, rendered as wordcloud images
    # (age has significant features in this sample; gender and is_student do not)
    run dla_controls_wordclouds "${DLATK[@]}" -d $DB -t $T -c $G -f "$NGRAM_FEAT_OCC" \
        --outcome_table $OT --group_freq_thresh $GFT \
        --outcomes age --controls gender \
        --correlate --tagcloud --make_wordclouds \
        --output_name "$OUT/dla_controls"
    check dla_controls_wordclouds "tagcloud and wordcloud images written" \
        eval 'files_exist "$OUT/dla_controls_tagcloud.txt" && dir_nonempty "$OUT/dla_controls_tagcloud_wordclouds"'

    # Wordclouds for every topic in a lexicon (a 10-topic slice of the Facebook topics, to keep it short)
    mysql "$DB" -e "DROP TABLE IF EXISTS $SMALL_TOPIC_LEX;
        CREATE TABLE $SMALL_TOPIC_LEX LIKE dlatk_lexica.$TOPIC_FREQ;
        INSERT INTO $SMALL_TOPIC_LEX SELECT * FROM dlatk_lexica.$TOPIC_FREQ
            WHERE category IN ('0','1','2','3','4','5','6','7','8','9');"
    run topic_wordclouds "${DLATK[@]}" -d $DB -t $T -c $G --lexicondb $DB --topic_lexicon $SMALL_TOPIC_LEX \
        --make_all_topic_wordclouds --tagcloud_colorscheme blue \
        --output_name "$OUT/topic_wordclouds"
    check topic_wordclouds "topic wordcloud images written" dir_nonempty "$OUT/topic_wordclouds_alltopics"

    # Prediction with the recommended feature selection pipeline
    run regression_magic_sauce "${DLATK[@]}" -d $DB -t $T -c $G -f "$TOPIC_FEAT" "$NGRAM_FEAT_OCC" \
        --outcome_table $OT --group_freq_thresh $GFT --outcomes age \
        --nfold_test_regression --model ridgecv --folds 10 --feature_selection magic_sauce --csv \
        --output_name "$OUT/regression_magic_sauce"
    check regression_magic_sauce "regression csv written" \
        files_exist "$OUT/regression_magic_sauce.accuracy_data.csv"

    run classification_magic_sauce "${DLATK[@]}" -d $DB -t $T -c $G -f "$TOPIC_FEAT" "$NGRAM_FEAT_OCC" \
        --outcome_table $OT --group_freq_thresh $GFT --outcomes is_student \
        --nfold_test_classifiers --model lr --folds 10 --feature_selection magic_sauce --csv \
        --output_name "$OUT/classification_magic_sauce"
    check classification_magic_sauce "classification csv written" \
        files_exist "$OUT/classification_magic_sauce.accuracy_data.csv"

    # Message cleaning: each creates a new message table derived from the scratch table
    if "$PYTHON_BIN" -c "import langid" 2>/dev/null; then
        run language_filter "${DLATK[@]}" -d $DB -t $T -c $G --language_filter en --clean_messages
        check language_filter "${T}_en populated" table_nonempty "${T}_en"
    else
        skip language_filter "python module langid not installed"
    fi

    run deduplicate "${DLATK[@]}" -d $DB -t $T -c $G --deduplicate
    check deduplicate "${T}_dedup populated" table_nonempty "${T}_dedup"

    run spam_filter "${DLATK[@]}" -d $DB -t $T -c $G --spam_filter
    check spam_filter "${T}_nospam populated" table_nonempty "${T}_nospam"

    # Transformer embeddings (one layer, mean per message then per user) on a 20-user,
    # 5-message slice of the scratch table so it stays quick on CPU
    mysql "$DB" -e "DROP TABLE IF EXISTS $EMB_T; CREATE TABLE $EMB_T LIKE $T;
        INSERT INTO $EMB_T SELECT m.* FROM $T m
        JOIN (SELECT DISTINCT $G FROM $T ORDER BY $G LIMIT 20) u USING ($G)" && cap_messages $EMB_T 5
    run transformer_emb "${DLATK[@]}" -d $DB -t $EMB_T -c $G \
        --add_emb_feat --emb_model $EMB_MODEL --emb_layers 5 --emb_table_name distilroberta
    check transformer_emb "768-dimension embedding feature table populated" \
        eval '[ "$(num_feats "$EMB_FEAT")" = 768 ]'

    # Sentence-transformer embeddings (self-contained; builds and drops its own tables)
    echo "=== sentence_transformer"
    if bash "$ROOT/tests/sentence_transformer.sh" > "$OUT/sentence_transformer.log" 2>&1; then
        if grep -q "^SKIP" "$OUT/sentence_transformer.log"; then
            SKIPPED+=("sentence_transformer")
            echo "    $(grep -m1 '^SKIP' "$OUT/sentence_transformer.log")"
        else
            PASSED+=("sentence_transformer")
            echo "    PASS"
        fi
    else
        FAILED+=("sentence_transformer")
        echo "    FAIL: see $OUT/sentence_transformer.log"
    fi
fi

[ "$KEEP" -eq 1 ] || drop_scratch

echo
echo "Passed (${#PASSED[@]}): ${PASSED[*]}"
echo "Failed (${#FAILED[@]}): ${FAILED[*]}"
echo "Skipped (${#SKIPPED[@]}): ${SKIPPED[*]}"
echo "Logs and outputs: $OUT"
[ ${#FAILED[@]} -eq 0 ]
