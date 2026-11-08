.. _fwflag_create_reliability_table:
==========================
--create_reliability_table
==========================
Switch
======

--create_reliability_table

Description
===========

Writes the reliability of every feature of an n-gram table, across groups and for each group, to a table named like the feature table with *relf$* in place of *feat$* (e.g. *relf$1gram$msgs$user_id*).

Argument and Default Value
==========================

None.

Details
=======

Reliabilities are calculated as for :doc:`fwflag_feat_reliability_filter`: split-half across groups, Spearman-Brown corrected and averaged over 10 random splits, or from a negative binomial model when most groups have one message (e.g. -c message_id). The table has a row for each row (group, feature) of the feature table whose group has at least :doc:`fwflag_group_freq_thresh` words:

* *value*: the feature's reliability across groups, the same in each of its rows (what :doc:`fwflag_feat_reliability_filter` compares to :doc:`fwflag_set_reliability`). It is negative when a feature's halves agree less than chance would, so the spread of the negative values shows how far chance alone moves reliabilities.
* *group_norm*: the reliability of that group's relative frequency of the feature. Sampling noise shrinks as 1/N with the group's N words (n-grams of the feature's size), and the across-group reliability is that of a group with the harmonic mean N of the groups, so a group gets the Spearman-Brown projection rel / (rel + (1 - rel) H / N), where H is the harmonic mean. Under the negative binomial model this is exactly (p^2/a) / (p^2/a + p/N). Negative reliabilities give 0. This shows how reliability changes across groups with few to many words (e.g. users with 500 vs 50,000 words); it is an approximation when groups' messages differ much in length.

Like the feature table, it is sparse: groups that never used a feature have no row for it; their reliability follows from *value* and the group's total in the feature table's meta table, by the formula above.

Later actions in the same command still use the feature table given by :doc:`fwflag_f`; with :doc:`fwflag_feat_reliability_filter` in the same command, the reliability table is of the table being filtered (not of the filtered one).

Other Switches
==============

Required Switches:

* :doc:`fwflag_f` an n-gram table (e.g. *feat$1gram...*, *feat$1to3gram...*)
* :doc:`fwflag_t`, :doc:`fwflag_c` the message table and group field the table was extracted from

Optional Switches:

* :doc:`fwflag_group_freq_thresh`

Example Commands
================
.. code-block:: bash

 dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \
   -f 'feat$1gram$msgs$user_id' --create_reliability_table
 # creates relf$1gram$msgs$user_id
