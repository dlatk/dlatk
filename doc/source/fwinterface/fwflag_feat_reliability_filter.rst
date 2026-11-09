.. _fwflag_feat_reliability_filter:
=========================
--feat_reliability_filter
=========================
Switch
======

--feat_reliability_filter

Description
===========

Keeps the features of an n-gram or lexicon (e.g. topic) table that are measured reliably at the group level (-c), i.e. whose value in a group mostly reflects the group rather than which messages it happened to write.

Argument and Default Value
==========================

None. The minimum reliability is set by :doc:`fwflag_set_reliability` (default 0.30).

Details
=======

Each group's messages are split at random into two halves with about as many words each, written as two copies of the message table (-t with *_sh0a* and *_sh0b* appended, dropped when done). The feature table's features are extracted from each half the way the table was made, with :doc:`fwflag_add_ngrams` or, for a lexicon table, :doc:`fwflag_add_ngrams` -n 1 then :doc:`fwflag_add_lex_table` (weighted when the table name says *_w*). The feature's *group_norm* in one half is correlated, across groups, with that in the other half, and the correlation is Spearman-Brown corrected, 2r / (1 + r), to give the reliability of the group_norm from all of the group's messages. This is averaged over random splits, which approaches coefficient alpha with each group's messages as items. A feature's reliability varies from split to split by about (1 - rel^2) / sqrt(G) over G groups, as a correlation does, so the number of splits is the fewest that keep the standard error of the average below 0.1 even when rel is 0: one split with 100 or more groups (that have at least two messages), more with fewer groups, up to 6; one when most groups have only two messages, since their splits are all the same. With two or more splits, another is added while the measured standard error (the median over features) is above 0.1, up to 6. Features with at least :doc:`fwflag_set_reliability` are kept, in a new table whose name ends in *$rel* and the threshold (e.g. *$rel0_3*).

Reliability is the share of a feature's variance across groups that is not sampling noise. A feature's correlation with an outcome is attenuated by the square root of its reliability, so unreliable features rarely reach significance in DLA and add mostly noise to prediction. Reliability depends on how many words each group has (expected count = rate x words), so the same feature can be reliable for users with thousands of words and unreliable for user-days or counties with few. Messages, not words, are the units split, so words used in bursts (several times in one message) are not treated as more reliable than they are.

How the table was made is read from its name: *feat$<n>gram$...*, *feat$<a>to<b>gram$...* (optionally *Up*) or *feat$cat_<lexicon>[_w]$...$1gra*, extracted from -t and -c, optionally followed by :doc:`fwflag_feat_occ_filter` or --feat_reliability_filter suffixes (e.g. *$0_05*, *$46*, *$rel0_3*), which only remove features. Other tables (e.g. value-transformed, smoothed, collocation or renamed tables) are refused. For a sample of groups, the two halves' values must add up to the feature table's (exactly for counts, within rounding for weighted lexicons), or nothing is calculated, since the table was then made some other way (e.g. from another message table or a filtered word table). Groups with fewer than two messages are left out of the split-half estimate.

When most groups have only one or two messages (e.g. -c message_id, or user-days where most users wrote once or twice a day), messages cannot be split, or every split of a group is the same and its halves can be far from equal, and an n-gram table's reliability is calculated parametrically (a lexicon table's is not calculated), with a warning, from a negative binomial (gamma-Poisson) model fit to the feature table, as :doc:`fwflag_smooth_feats` nb fits it. A feature with background rate p and dispersion a has true rates that vary across groups with variance p^2/a, and a group with N words adds sampling variance p/N, so reliability across groups is (p^2/a) / (p^2/a + p mean(1/N)), the same quantity the split-half estimates. This assumes words are independent draws, so words used in bursts look more reliable than they are, which leads to a slight overestimate of reliability for common words. It needs only the feature table and its meta table of n-gram totals, so it is fast.

Groups below :doc:`fwflag_group_freq_thresh` are left out. Run it after :doc:`fwflag_feat_occ_filter`. Each split extracts the features from all of the messages once (both halves), so it takes 1 to 6 times as long as extracting the table (once with 100 or more groups).

Other Switches
==============

Required Switches:

* :doc:`fwflag_f` an n-gram or lexicon table (e.g. *feat$1gram...*, *feat$1to3gram...*, *feat$cat_met_a30_2000_cp_w...$1gra*)
* :doc:`fwflag_t`, :doc:`fwflag_c` the message table and group field the table was extracted from

Optional Switches:

* :doc:`fwflag_set_reliability`
* :doc:`fwflag_group_freq_thresh`

Example Commands
================
.. code-block:: bash

 dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \
   -f 'feat$1to3gram$msgs$user_id$0_05' --feat_reliability_filter
 # creates feat$1to3gram$msgs$user_id$0_05$rel0_3
 dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \
   -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$1gra' --feat_reliability_filter --set_reliability 0.5
 # creates feat$cat_met_a30_2000_cp_w$msgs$user_id$1gra$rel0_5
