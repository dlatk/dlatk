# Unit Testing

`unit_tests.sh` runs the core DLATK pipeline (n-grams, topics, DLA, regression, classification) plus
extended tests, using only the data distributed with DLATK. It is built from the helper scripts below,
which can also be run on their own.

All scripts run the repo's own `dlatkInterface.py` (found relative to this `tests/` folder), not an
installed copy, so they test the code you are editing.

## Running the unit tests

Load the tutorial data once:

```bash
mysql -e 'CREATE DATABASE dla_tutorial'; cat dlatk/data/dla_tutorial.sql | mysql dla_tutorial
mysql -e 'CREATE DATABASE dlatk_lexica'; cat dlatk/data/dlatk_lexica.sql | mysql dlatk_lexica
```

Then, from the repo root:

```bash
bash tests/unit_tests.sh        # all tests (~15 minutes on CPU)
bash tests/unit_tests.sh -c     # core tests only
bash tests/unit_tests.sh -k     # keep the scratch tables afterwards...
bash tests/unit_tests.sh -s     # ...so a rerun can skip sampling and n-gram extraction (~8 minutes)
bash tests/unit_tests.sh -h     # all options (-n users, -m messages per user, -p p_occ, -o output folder)
```

The tests run on `dla_tutorial.msgs_unittest`, a fixed random sample of 150 users capped at 30 messages
each, so `msgs` is never modified. Every test ends as **PASS**, **FAIL** or **SKIP**; FAIL and SKIP give a
reason and the log to look at, and the script exits non-zero if any test failed. dlatk can exit 0 after an
error, so each test also checks for the tables or files its step should produce.

See the bottom of `dev_instructions.txt` for what each test covers.

## Helper scripts

Each runs one part of the pipeline; options match their `dlatkInterface.py` counterparts, and lists
(feature tables, outcomes, controls) are passed as one quoted, space-separated argument. Run any of them
with `-h` for usage, e.g.

    bash tests/correlate.sh -h

| Script | Runs |
|---|---|
| `add_ngrams.sh` | `--add_ngrams` and `--combine_feat_tables`, then optional `--feat_occ_filter` (`--set_p_occ`) and `--feat_colloc_filter` (`--set_pmi_threshold`); `--no_extract` runs only the filters |
| `add_lex_table.sh` | `--add_lex_table --weighted_lexicon` |
| `add_embeddings.sh` | `--add_emb_feat` (default model `distilroberta-base`) |
| `correlate.sh` | `--correlate --rmatrix --csv --sort` with tagclouds (topic tagclouds with `--topic_lexicon`), optional `--controls`, `--categories_to_binary`, `--make_wordclouds` |
| `regression.sh` | `--nfold_test_regression` (default `--model ridgecv --folds 10`), optional `--feature_selection` |
| `classification.sh` | `--nfold_test_classifiers` (default `--model lr --folds 10`), optional `--categories_to_binary`, `--feature_selection` |
| `clean_messages.sh` | `--language_filter <LANG> --clean_messages`, `--deduplicate`, `--spam_filter` (any combination) |
| `make_all_topic_wordclouds.sh` | `--make_all_topic_wordclouds` for a topic lexicon |
| `sentence_transformer.sh` | self-contained sentence-transformer embedding test (`--add_sent_emb_feat`) |

`common.sh` holds the setup the helpers share. Example:

```bash
bash tests/correlate.sh -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id' \
  --outcome_table blog_outcomes --outcomes 'age gender' --output_name /tmp/dla_out
```
