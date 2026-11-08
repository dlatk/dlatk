.. _fwflag_feat_reliability_filter:
=========================
--feat_reliability_filter
=========================
Switch
======

--feat_reliability_filter

Description
===========

Keeps the features of an n-gram table that are measured reliably at the group level (-c), i.e. whose relative frequency in a group mostly reflects the group rather than which messages it happened to write.

Argument and Default Value
==========================

None. The minimum reliability is set by :doc:`fwflag_set_reliability` (default 0.30).

Details
=======

Each group's messages are split at random into two halves with about as many words each. The feature's relative frequency in one half is correlated, across groups, with that in the other half, and the correlation is Spearman-Brown corrected, 2r / (1 + r), to give the reliability of the relative frequency from all of the group's messages. This is averaged over 10 random splits, which approaches coefficient alpha with each group's messages as items. Features with at least :doc:`fwflag_set_reliability` are kept, in a new table whose name ends in *$rel* and the threshold (e.g. *$rel0_3*).

Reliability is the share of a feature's variance across groups that is not sampling noise. A feature's correlation with an outcome is attenuated by the square root of its reliability, so unreliable features rarely reach significance in DLA and add mostly noise to prediction. Reliability depends on how many words each group has (expected count = rate x words), so the same feature can be reliable for users with thousands of words and unreliable for user-days or counties with few. Messages, not words, are the units split, so words used in bursts (several times in one message) are not treated as more reliable than they are.

The filter counts n-grams from the message table (-t) the same way :doc:`fwflag_add_ngrams` does, and warns if a sample of groups does not match the feature table (e.g. a table built from collocations or another message table). Groups with fewer than two messages are left out of the split-half estimate.

When most groups have only one message (e.g. -c message_id, or user-days where most users wrote once a day), messages cannot be split, and reliability is calculated parametrically, with a warning, from a negative binomial (gamma-Poisson) model fit to the feature table, as :doc:`fwflag_smooth_feats` nb fits it. A feature with background rate p and dispersion a has true rates that vary across groups with variance p^2/a, and a group with N words adds sampling variance p/N, so reliability across groups is (p^2/a) / (p^2/a + p mean(1/N)), the same quantity the split-half estimates. This assumes words are independent draws, so words used in bursts look more reliable than they are, which leads to a slight overestimate of reliability for common words. It needs only the feature table and its meta table of n-gram totals, so it is fast.

Groups below :doc:`fwflag_group_freq_thresh` are left out. Run it after :doc:`fwflag_feat_occ_filter`; with tens of thousands of features, it takes about as long as extracting the n-grams.

Other Switches
==============

Required Switches:

* :doc:`fwflag_f` an n-gram table (e.g. *feat$1gram...*, *feat$1to3gram...*)
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
