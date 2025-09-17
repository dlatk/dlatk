# Unit Testing
Above collection of test scripts are divided based on the components of DLA pipeline, and named after their respective switches in DLATK - 
 
 - Data cleaning - `clean_messages.sh`
 - N-gram extraction - `add_ngrams.sh` 
 - Lexicon feature extraction - `add_lex_table.sh` 
 - Feature correlations against outcomes - `correlate.sh`
 - Regression - `regression.sh` 
 - Classification - `classification.sh` 
 - Topic visualization - `make_all_topic_wordclouds.sh`

And a master script called `unit_tests.sh` that tests all of this components with defaults.

## Flags
Each script takes in one or more parameters (flags) that are synonymous with their counterparts in DLATK. You can know about them using the help switch `-h`. For example, 

    bash add_ngrams.sh -h
gives

    Usage - bash add_ngrams.sh -d <DB> -t <TABLE> -c <GROUP_FIELD> --group_freq_thresh <GFT> --set_p_occ <OCC> --set_pmi_threshold <PMI>

## Master script (`unit_tests.sh`) example:

```bash
bash unit_tests.sh \
  -e mysql \
  -d dla_tutorial \
  -t msgs \
  -c user_id \
  -f 'feat$1gram$msgs$user_id' \
  -l lda50_cp \
  --freq_table lda50_freq_t50ll \
  --outcome_table toy_outcomes \
  --classification_outcome y_bin \
  --regression_outcome y_real \
  --output_folder unit_out
```

NOTE: this is when we have all table generated in realated flag 



