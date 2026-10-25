# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DLATK is an end-to-end social media text analysis toolkit. The single entry point is `dlatkInterface.py`; nearly all functionality is exposed as CLI flags. Primary backend is MySQL; SQLite is also supported (pass a `.csv` path as `-t` to auto-import and use SQLite).

## Setup

```sh
python setup.py install
# Load tutorial data into MySQL:
mysql -e 'CREATE DATABASE dla_tutorial'
cat dlatk/data/dla_tutorial.sql | mysql dla_tutorial
mysql -e 'CREATE DATABASE dlatk_lexica'
cat dlatk/data/dlatk_lexica.sql | mysql dlatk_lexica
```

MySQL credentials are read from `~/.my.cnf`. To import CSV files into MySQL use `usefulScripts/csv2mySQLv2.py`:

```sh
python3 usefulScripts/csv2mySQLv2.py FILE DATABASE TABLENAME '(col_def)' IGNORE_LINES
# Example:
python3 usefulScripts/csv2mySQLv2.py dlatk/data/msgs100u.csv dla_tutorial msgs \
  '(message_id INT, user_id INT, created_date VARCHAR(64), message TEXT)' 1
```

## Running Tests

Tests live in `tests/` as shell scripts that run the repo's own `dlatkInterface.py`. The suite needs only the
tutorial data (`dla_tutorial`, `dlatk_lexica`) and builds everything else itself on a scratch sample
(`dla_tutorial.msgs_unittest`: 150 users x 30 messages), so `msgs` is never modified:

```sh
bash tests/unit_tests.sh          # all tests, ~15 min on CPU; ends with PASS/FAIL/SKIP (with reasons)
bash tests/unit_tests.sh -c       # core tests only (n-grams, topics, DLA, ridge, lr)
bash tests/unit_tests.sh -k       # keep scratch tables, then rerun with -s to skip n-gram extraction (~8 min)
```

`unit_tests.sh` is built from helper scripts that can be run individually (`-h` for usage; lists are one
quoted, space-separated argument), e.g.:

```sh
bash tests/add_ngrams.sh -d dla_tutorial -t msgs -c user_id -n '1 2' --set_p_occ 0.05
bash tests/correlate.sh -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id' --outcome_table blog_outcomes --outcomes 'age gender' --output_name /tmp/corr_out
bash tests/regression.sh -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id' --outcome_table blog_outcomes --outcomes age --output_name /tmp/reg_out
```

See `tests/README.md` and the bottom of `dev_instructions.txt` for details. dlatk can exit 0 after an error,
so tests check for the tables/files each step should produce rather than relying on exit codes.

## Typical DLA Pipeline

**Step 0 — Prepare a personal copy of the message table** (replace `xxx` with your initials):

```sql
CREATE TABLE msgs_xxx LIKE msgs;
INSERT INTO msgs_xxx SELECT * FROM msgs;
```

**Step 1 — Feature extraction:**

```sh
# 1-3 grams
python3 dlatkInterface.py -d dla_tutorial -t msgs_xxx -g user_id \
  --add_ngrams -n 1 2 3 --combine_feat_tables 1to3gram
# Filter rare features (used by <5% of groups or groups with <500 words)
python3 dlatkInterface.py -d dla_tutorial -t msgs_xxx -g user_id \
  -f 'feat$1to3gram$msgs_xxx$user_id' \
  --feat_occ_filter --set_p_occ 0.05 --group_freq_thresh 500
# Weighted lexicon (Facebook topics)
python3 dlatkInterface.py -d dla_tutorial -t msgs_xxx -g user_id \
  --add_lex_table -l met_a30_2000_cp --weighted_lexicon
```

**Step 2 — DLA / correlation:**

```sh
python3 dlatkInterface.py -d dla_tutorial -t msgs_xxx -g user_id \
  -f 'feat$1to3gram$msgs_xxx$user_id$0_05' \
  --outcome_table blog_outcomes --group_freq_thresh 500 \
  --outcomes age gender --output_name /tmp/dla_out \
  --correlate --rmatrix --csv --sort --tagcloud --make_wordclouds
```

**Step 3 — Prediction:**

```sh
# Regression (continuous)
python3 dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id \
  -f 'feat$cat_met_a30_2000_cp_w$msgs_xxx$user_id$1gra' \
  --outcome_table blog_outcomes --group_freq_thresh 500 \
  --outcomes age --output_name /tmp/pred_out \
  --nfold_test_regression --model ridgecv --folds 10

# Classification (binary)
python3 dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id \
  -f 'feat$cat_met_a30_2000_cp_w$msgs_xxx$user_id$1gra' \
  --outcome_table blog_outcomes --group_freq_thresh 500 \
  --outcomes gender --output_name /tmp/class_out \
  --nfold_test_classifiers --model lr --folds 10
```

## Feature Table Naming

`feat$<type>$<msg_table>$<group_col>[$<options>]`

| Suffix | Meaning |
|---|---|
| `16to16` | relative frequency (default) |
| `16to1` | boolean |
| `16to8` | Anscombe transform |
| `16to4` | sqrt transform |
| `0_05` | occurrence filter (5% threshold) |
| `46` | occurrence filter by group count (`--set_p_occ 46`, or the count chosen by `--set_p_occ auto`) |
| `pmi3_0` | PMI threshold 3.0 |

Examples:
- `feat$1gram$msgs$user_id` — raw 1-grams grouped by user_id
- `feat$1to3gram$msgs$user_id$0_05` — 1–3 grams, 5% occurrence filter
- `feat$cat_met_a30_2000_cp_w$msgs$user_id$1gra` — weighted Facebook topic features

When building and later applying a pickled model, the `-f` feature table list must be in the **exact same order**.

## Key Flags

| Flag | Meaning |
|---|---|
| `-d` | database |
| `-t` | message table (or `.csv` path for SQLite auto-import) |
| `-c` / `-g` | group/correlation field |
| `-f` | feature table(s) |
| `-e` | engine: `mysql` (default) or `sqlite` |
| `--group_freq_thresh` | min words per group (500 typical for users) |
| `--outcome_table` | table with outcome variables |
| `--outcomes` | outcome columns to analyze |
| `--weighted_lexicon` | use weighted (vs. unweighted) lexicon scoring |
| `--feature_selection magic_sauce` | recommended feature selection pipeline |
| `--save_model` / `--load_models` | pickle model to/from `--picklefile` |

## SQLite / CSV Quick Start (no MySQL needed)

Passing a `.csv` path as `-t` triggers auto-import into SQLite under `~/sqlite_data/`:

```sh
python3 dlatkInterface.py -e sqlite \
  -t dlatk/data/msgs100u.csv -c user_id \
  --add_ngrams -n 1
```

## Architecture

`DLAWorker` (`dlatk/dlaWorker.py`) is the base class; it owns the `DataEngine` and `QueryBuilder`. All workers inherit from it. The database layer (`dlatk/database/dataEngine.py`) abstracts MySQL vs. SQLite. Constants and defaults are in `dlatk/dlaConstants.py`.

Key workers: `FeatureExtractor`, `FeatureGetter`, `FeatureRefiner`, `OutcomeAnalyzer`, `RegressionPredictor`, `ClassifyPredictor`, `TopicExtractor`, `MessageTransformer`, `SemanticsExtractor`.

## Commit Conventions

- Branch per feature: `git checkout -b branch_name`
- Prefix commit messages: `FIXED BUG: ...` or `ADDED FEATURE: ...`
- Document any new `--` CLI flags in `doc/`
