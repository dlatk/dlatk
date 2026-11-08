.. _fwflag_smooth_zero_gap:
=================
--smooth_zero_gap
=================
Switch
======

--smooth_zero_gap

Description
===========

With :doc:`fwflag_smooth_feats` dirichlet, nb, pln or zinb, smooths only the groups that used a feature: groups that never used it keep 0, so used and unused stay apart.

Argument and Default Value
==========================

None

Details
=======

Smoothing gives every group that never used a feature a small rate, which differs across groups (it depends on each group's word count). Whether a group used a word at all can carry much of its signal, especially when groups have few words; with a zero gap that signal is kept and only the rates of groups that used the feature are smoothed. Groups that never used it get 0, which is below every smoothed rate (and its square root, with :doc:`fwflag_smooth_sqrt`). With :doc:`fwflag_smooth_log` they get the log of half the feature's smallest smoothed rate among the groups that used it, a gap below them as 0 is below the smallest rate.

The table name gets a *zg* before the method (e.g. *$zgnb*, *$zgnbL*, *$zgnbS*). Without :doc:`fwflag_smooth_log`, unused groups are 0 and are left out of the table as in an unsmoothed one, so the table is no larger than the original; with it, the table is dense. The *hurdle* method keeps the same two parts in two separate tables.

Other Switches
==============

Required Switches:

* :doc:`fwflag_smooth_feats` dirichlet, nb, pln or zinb

Optional Switches:

* :doc:`fwflag_smooth_sqrt` or :doc:`fwflag_smooth_log`

Example Commands
================
.. code-block:: bash

 dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \
   -f 'feat$1gram$msgs$user_id$0_05' --smooth_feats nb --smooth_sqrt --smooth_zero_gap
 # creates feat$1gram$msgs$user_id$0_05$zgnbS
