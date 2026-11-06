.. _fwflag_smooth_feats:
==============
--smooth_feats
==============
Switch
======

--smooth_feats METHOD

Description
===========

Creates a copy of an n-gram feature table whose group_norm is a smoothed estimate of each feature's rate, instead of the maximum likelihood estimate (count / total) used by default.

Argument and Default Value
==========================

One of: dirichlet, nb, pln, zinb, logodds_idp, logodds, hurdle. No default.

Details
=======

A group's group_norm (count of the n-gram over its total n-grams) is noisy when the group has few words: a word used once in a 20-word group gets the same 0.05 as a word used 50 times in 1,000 words, and every unused word gets 0. Smoothing pulls each group's estimate toward the feature's rate in the whole corpus, by an amount that shrinks as the group's word count grows. Groups with many words barely change, so this matters most for short groups (messages, user-days, users with few posts). Note that the amount of smoothing must vary across groups to change results: smoothing every group by the same proportion is a linear transform and leaves correlations and standardized models unchanged.

For a feature with count *c* in a group with *N* total n-grams (from the meta table), and corpus rate *p* (its count over all groups' totals, for groups above :doc:`fwflag_group_freq_thresh`):

* **dirichlet**: *(c + mu p) / (N + mu)*, as if each group had *mu* more words spread like the corpus. By default *mu* is estimated for each group by leave-one-out likelihood: groups whose language is typical of the corpus get a larger *mu* (more smoothing), distinctive groups a smaller one. Groups with under 50 words have their *mu* pulled toward the corpus-wide estimate. With :doc:`fwflag_smooth_log`, every group uses the corpus-wide estimate: on the log scale a never-used word's value is driven by its group's *mu*, which would add a factor shared by all features (and correlated with whatever the group's language reflects). :doc:`fwflag_smooth_mu` sets one constant instead.
* **nb**: negative binomial (gamma-Poisson) empirical Bayes, *(c + a) / (N + a/p)*. The prior strength comes from each feature's dispersion *a*, estimated from how much its rate varies across groups beyond chance: features whose variation is mostly sampling noise (e.g. function words) are smoothed strongly, bursty topical features little.
* **pln**: Poisson-log-normal, the same as nb but with a log-normal prior, which fits user- and county-level rates better (Almodaresi et al., 2017). Computed in closed form (a Laplace approximation using the Lambert W function).
* **zinb**: zero-inflated negative binomial: some groups never use the feature at all. Each feature's share of such groups is fit by EM; a zero then counts partly as "never uses it".
* **logodds_idp**: z-scored log-odds of the feature in the group versus the rest of the corpus, with an informative Dirichlet prior (Monroe, Colaresi & Quinn, 2008). Uses *mu* as dirichlet does.
* **logodds**: the same with an uninformative prior (0.5 added to every count), as a baseline.
* **hurdle**: two tables, *TABLE$hurB* (1 if the group used the feature, else 0) and *TABLE$hurL* (the nb log rate where it was used, the feature's mean log rate elsewhere). Use both with :doc:`fwflag_f` in prediction.

With :doc:`fwflag_smooth_log`, dirichlet, nb, pln and zinb store the posterior expected log rate, which suits rates that are close to log-normal across groups.

The new table is named after the original plus a suffix (*$dir*, *$dir500* for a constant *mu*, *$nb*, *$pln*, *$zinb*, *$loi*, *$lo*, *$hurL*/*$hurB*, with *L* added for :doc:`fwflag_smooth_log`). Its value column keeps the raw count. Because no smoothed rate is 0, the table is dense: one row per feature per group, with a warning if that is more than 10 times the rows of the original. Smoothing should therefore come last: filter features first with :doc:`fwflag_feat_occ_filter` (which counts rows, so it should not be run on a smoothed table).

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t`, :doc:`fwflag_c`
* :doc:`fwflag_f` an n-gram table (its meta table supplies the totals)

Optional Switches:

* :doc:`fwflag_smooth_mu`
* :doc:`fwflag_smooth_log`
* :doc:`fwflag_group_freq_thresh`
* :doc:`fwflag_feat_occ_filter` with :doc:`fwflag_set_p_occ` (run in the same command, it filters first)

Example Commands
================
.. code-block:: bash

 # filter 1grams, then smooth them with per-user dirichlet priors and store log rates
 dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \
   -f 'feat$1gram$msgs$user_id' --feat_occ_filter --set_p_occ 0.05 --smooth_feats dirichlet --smooth_log
 # creates feat$1gram$msgs$user_id$0_05$dirL
