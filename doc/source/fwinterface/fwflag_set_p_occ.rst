.. _fwflag_set_p_occ:
===========
--set_p_occ
===========
Switch
======

--set_p_occ <pocc_thresh>

Description
===========

Sets the threshold for :doc:`fwflag_feat_occ_filter`: how many groups must use a feature for it to be kept.

Argument and Default Value
==========================

pocc_thresh can be:

* a fraction below 1: keep features used by at least this fraction of the groups (those above :doc:`fwflag_group_freq_thresh` words). The new table ends in the fraction, e.g. ``$0_05``.
* a number of 1 or more: keep features used by at least this many groups. The new table ends in the count, e.g. ``$46``.
* ``auto``: choose the number of groups from the number of groups, N, keeping a feature count suited to DLA (see below).
* ``auto_prediction``: a more lenient ``auto`` for features used in prediction.

Default: 0.01

Details
=======

**auto.** A feature must be used by at least k = max(5, 0.75·√N) groups, and at most 5·N^0.75 features are kept; if more pass, k is raised until they fit. The chosen k is printed and names the table (e.g. ``$46``), so ``--set_p_occ 46`` reproduces it.

Why these rules: how reliably a feature is measured within a group depends on how many words the group has, not on how many groups there are. Since n-gram use is zero-inflated (each group has its own vocabulary and simply never uses most words), a feature used by k groups is estimated from those k groups, and the smallest effect DLA can detect on it shrinks with √k. Requiring k to grow with √N keeps rare, noisy features out while still admitting more features as groups gain words, because then more groups use each feature. The cap reflects that every extra feature makes the multiple-test correction in DLA harder to pass: with few words per group the minimum number of groups sets the count, and with many words per group the cap does. This is the intended behavior: once groups have many words, each group's feature values are already measured reliably, and what limits how many features DLA can usefully test is the number of groups, which supply the variation across groups that correlations are estimated from and the statistical power to survive correction for testing more features.

**auto_prediction** essentially halves the criteria: it requires half as many groups (k/2) and keeps up to 10·N features (10 per group). Prediction models are much more admissible of features, since a feature does not have to be significant on its own to help a prediction. The cap grows with N rather than N^0.75 because what limits a prediction model is how many features it can estimate from N groups without overfitting, rather than a multiple-test correction. The cap still leans toward fewer features (bias over variance), so that an automatic choice is unlikely to overfit. It warns above 200,000 features.

Every table is filtered the same way; k is never raised above N. The occurrence filter is designed for frequency- or count-based features (n-grams, lexica and topics, whose zeros are left out of the table), so the filter warns when every feature is used by every group, as with embeddings.

**Estimating how many n-gram features are kept.** For 1-2grams, the number of features used by at least a fraction p of groups is approximately

    features ≈ W / (14.8 · λ^1.28),  where λ = −ln(1 − p)

and W is the average number of words per group (fit on the tutorial blogs for W of about 350 to 19,000 words; within about 25%). It grows in proportion to W: doubling the words per group roughly doubles the features that pass a given threshold. For ``auto`` and large N, p ≈ 0.75/√N, giving approximately W · N^0.64 / 10 features before the cap. Examples (rounded estimates; N = 100,000 is extrapolated):

=========  =====  =====  =====  =======  ========
auto       W=30   W=200  W=500  W=2,000  W=20,000
=========  =====  =====  =====  =======  ========
N=50       28     94     94     94       94
N=500      149    529    529    529      529
N=5,000    653    3.0k   3.0k   3.0k     3.0k
N=100,000  4.5k   28k    28k    28k      28k
=========  =====  =====  =====  =======  ========

================  =====  =====  =====  =======  ========
auto_prediction   W=30   W=200  W=500  W=2,000  W=20,000
================  =====  =====  =====  =======  ========
N=50              71     472    500    500      500
N=500             339    2.3k   5.0k   5.0k     5.0k
N=5,000           1.6k   11k    26k    50k      50k
N=100,000         11k    73k    183k   733k     1.0M
================  =====  =====  =====  =======  ========

Other Switches
==============

Required Switches:

* :doc:`fwflag_feat_occ_filter`

Example Commands
================

.. code-block:: bash

	# keep features used by at least 5% of users with 500+ words
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1to3gram$msgs$user_id' --feat_occ_filter --set_p_occ 0.05 --group_freq_thresh 500

	# let dlatk choose the threshold for DLA
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1to3gram$msgs$user_id' --feat_occ_filter --set_p_occ auto --group_freq_thresh 500
