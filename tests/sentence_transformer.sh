#!/usr/bin/env bash
# DLATK Sentence-Transformers integration smoke test (uses existing DB: dla_tutorial)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DLATK="${ROOT}/dlatkInterface.py"

# Use your conda env's python if provided; otherwise fall back to python3
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "SKIP: python not found at PYTHON_BIN=$PYTHON_BIN"
  exit 0
fi

echo "[dlatk-sent-emb] starting (DB=dla_tutorial)"

# --- Preflight ---
if ! command -v mysql >/dev/null 2>&1; then
  echo "SKIP: mysql client not found"
  exit 0
fi

# Require Python >= 3.9 (avoid tokenizer overflow issues)
"$PYTHON_BIN" - <<'PY' || { echo "SKIP: Python < 3.9 (DLATK ST requires 3.9)"; exit 0; }
import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)
PY

if [[ ! -f "$DLATK" ]] && ! command -v dlatkInterface.py >/dev/null 2>&1; then
  echo "SKIP: dlatkInterface.py not found (run from repo root or install DLATK)"
  exit 0
fi

# MySQL connection (defaults to ~/.my.cnf if present)
MYSQL_HOST="${MYSQL_HOST:-localhost}"
MYSQL_PORT="${MYSQL_PORT:-3306}"
MYSQL_USER="${MYSQL_USER:-}"
MYSQL_PWD="${MYSQL_PWD:-}"  

DB="dla_tutorial"
TABLE="msgs_sts$RANDOM"

mysql_exec() {
  local SQL="$1"
  if [[ -n "$MYSQL_PWD" ]]; then
    mysql -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p"$MYSQL_PWD" -N -e "$SQL"
  else
    mysql -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -N -e "$SQL"
  fi
}

# Ensure we can USE dla_tutorial
mysql_exec "USE \`$DB\`; SELECT 1;" >/dev/null 2>&1 || {
  echo "FAIL: Cannot USE database $DB with user $MYSQL_USER"
  exit 1
}

# Clean up handler: drop our msgs table and any feat tables we created
cleanup() {
  mysql_exec "USE \`$DB\`; DROP TABLE IF EXISTS \`$TABLE\`, \`$TABLE\`_bert_trunc;" >/dev/null 2>&1 || true
  drop_feats_for() {
    local SRC="$1"
    while read -r FT; do
      [[ -z "${FT:-}" ]] && continue
      mysql_exec "USE \`$DB\`; DROP TABLE IF EXISTS \`$FT\`;" >/dev/null 2>&1 || true
    done < <(mysql_exec "SELECT table_name FROM information_schema.tables
                        WHERE table_schema='${DB}'
                          AND table_name LIKE CONCAT('feat\$','%','\$','${SRC}','\$','user_id','%');")
  }
  drop_feats_for "$TABLE"
  drop_feats_for "${TABLE}_bert_trunc"
}


trap cleanup EXIT

# Create tiny msgs table and insert pairs
mysql_exec "USE \`$DB\`;
  CREATE TABLE \`$TABLE\` (
    message_id BIGINT PRIMARY KEY,
    user_id VARCHAR(64),
    message TEXT
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"

# Add index on user_id (speeds up DLATK read & silences warning)
mysql_exec "USE \`$DB\`; ALTER TABLE \`$TABLE\` ADD INDEX (\`user_id\`);"

mysql_exec "USE \`$DB\`;
  INSERT INTO \`$TABLE\` (message_id,user_id,message) VALUES
   (1,'u1','A man is playing a guitar.'),
   (2,'u2','A person is playing a guitar.'),
   (3,'u3','A child is running through a field of grass.'),
   (4,'u4','A kid runs through the grass.'),
   (5,'u5','An airplane is flying in the sky.'),
   (6,'u6','A jet aircraft is soaring in the air.'),
   (7,'u7','A man is playing a guitar.'),
   (8,'u8','A woman is making a sandwich.'),
   (9,'u9','Two dogs are running through a field.'),
   (10,'u10','There are no animals outside.');"

# Models to test (BERT optional if tokenizer overflow persists)
MODELS=("sentence-transformers/stsb-distilroberta-base-v2")
MODELS+=("sentence-transformers/bert-base-nli-mean-tokens")
if [[ "${INCLUDE_MXBAI:-0}" == "1" ]]; then
  MODELS+=("mixedbread-ai/mxbai-embed-xsmall-v1")
fi

run_dlatk() {
  if [[ -f "$DLATK" ]]; then
    "$PYTHON_BIN" "$DLATK" "$@"
  else
    dlatkInterface.py "$@"
  fi
}

find_feat_table() {
  local logfile="$1"
  local src_table="$2"

  # Parse DLATK's line first
  local ft
  ft="$(awk -F' - ' '/^Feature table(s) - /{print $2}' "$logfile" | tail -n1)"
  if [[ -n "$ft" ]]; then
    echo "$ft"
    return
  fi

  # Fallback to information_schema
  mysql_exec "SELECT table_name
              FROM information_schema.tables
              WHERE table_schema='${DB}'
                AND table_name LIKE CONCAT('feat\$','%','\$','${src_table}','\$','user_id','%')
              ORDER BY create_time DESC
              LIMIT 1;"
}

check_model() {
  local MODEL="$1"
  echo "[run] $MODEL"

  # Choose which table to run on
  local RUN_TABLE="$TABLE"

  # Truncate messages to avoid tokenizer overflow (BERT only)
  if [[ "$MODEL" == "sentence-transformers/bert-base-nli-mean-tokens" ]]; then
    RUN_TABLE="${TABLE}_bert_trunc"
    local MAX_CHARS="${MAX_CHARS:-2000}"
    mysql_exec "USE \`$DB\`;
      DROP TABLE IF EXISTS \`${RUN_TABLE}\`;
      CREATE TABLE \`${RUN_TABLE}\` LIKE \`${TABLE}\`;
      INSERT INTO \`${RUN_TABLE}\` (message_id, user_id, message)
      SELECT message_id, user_id, LEFT(message, ${MAX_CHARS})
      FROM \`${TABLE}\`;
      ALTER TABLE \`${RUN_TABLE}\` ADD INDEX (\`user_id\`);"
  fi

  # capture DLATK stdout/stderr
  local LOG
  LOG="$(mktemp -t dlatk_log.XXXXXX)" || { echo "mktemp failed"; return 1; }

  if ! run_dlatk -d "$DB" -t "$RUN_TABLE" -c "user_id" \
        --messageid_field "message_id" --message_field "message" \
        --add_sent_emb_feat --emb_model "$MODEL" | tee "$LOG"
  then
    echo "FAIL: DLATK run failed for $MODEL"
    rm -f "$LOG"
    return 1
  fi

  local FEAT_TABLE=""
  FEAT_TABLE="$(find_feat_table "$LOG" "$RUN_TABLE" | tail -n1 || true)"
  rm -f "$LOG"

  if [[ -z "$FEAT_TABLE" ]]; then
    echo "FAIL: could not find feature table for $MODEL"
    return 1
  fi

  # pull vectors and score cosine similarities
  local TMP_TSV
  TMP_TSV="$(mktemp -t dlatk_vecs.XXXXXX)" || { echo "mktemp failed"; return 1; }

  mysql_exec "USE \`$DB\`; SELECT group_id, feat, value
              FROM \`$FEAT_TABLE\`
              WHERE group_id IN ('u1','u2','u3','u4','u5','u6','u7','u8','u9','u10');" > "$TMP_TSV"

  if ! grep -q . "$TMP_TSV"; then
    echo "FAIL: no feature rows returned from $FEAT_TABLE (check group_id filter vs -c user_id/message_id)"
    rm -f "$TMP_TSV"
    return 1
  fi

  "$PYTHON_BIN" - "$MODEL" "$TMP_TSV" <<'PY'
import sys, json, math
from collections import defaultdict
model, path = sys.argv[1], sys.argv[2]
vecs = defaultdict(dict)
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        gid, feat, val = line.rstrip("\n").split("\t")
        vecs[gid][feat] = float(val)
def dot(a,b): return sum(a[k]*b[k] for k in (a.keys() & b.keys()))
def norm(a):  return math.sqrt(sum(v*v for v in a.values())) or 1.0
def cos(a,b): return dot(a,b) / (norm(a)*norm(b))
high_pairs = [("u1","u2"), ("u3","u4"), ("u5","u6")]
low_pairs  = [("u7","u8"), ("u9","u10")]
high = [cos(vecs[i], vecs[j]) for (i,j) in high_pairs]
low  = [cos(vecs[i], vecs[j]) for (i,j) in low_pairs]
mh, ml = sum(high)/len(high), sum(low)/len(low)
sep = mh - ml
ok = (mh > 0.60) and (ml < 0.40) and (sep >= 0.25)
print(json.dumps({"model": model, "mean_high": mh, "mean_low": ml, "sep": sep, "ok": ok}))
sys.exit(0 if ok else 1)
PY
  local rc=$?
  rm -f "$TMP_TSV"

  if [[ $rc -eq 0 ]]; then
    echo "PASS: $MODEL"
  else
    echo "FAIL: $MODEL"
  fi
  return $rc
}

fail=0
for m in "${MODELS[@]}"; do
  check_model "$m" || fail=1
done

if [[ $fail -eq 0 ]]; then
  echo "[dlatk-sent-emb] done"
else
  echo "[dlatk-sent-emb] failed"
  exit 1
fi

