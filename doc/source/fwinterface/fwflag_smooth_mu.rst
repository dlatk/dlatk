.. _fwflag_smooth_mu:
===========
--smooth_mu
===========
Switch
======

--smooth_mu MU

Description
===========

A constant prior strength, in words, for :doc:`fwflag_smooth_feats` dirichlet and logodds_idp.

Argument and Default Value
==========================

A number of words. Default: none, *mu* is estimated for each group by leave-one-out likelihood (for dirichlet with :doc:`fwflag_smooth_log`, one corpus-wide estimate is used for every group).

Details
=======

Smoothing acts as if each group had *mu* more words spread like the whole corpus, so a group with *N* words keeps a weight of *N / (N + mu)* on its own counts. With this flag every group uses the same *mu*, and the table name records it (e.g. *$dir500*). Without it, groups whose language is typical of the corpus are smoothed more than distinctive ones.

Other Switches
==============

Required Switches:

* :doc:`fwflag_smooth_feats` dirichlet or logodds_idp

Example Commands
================
.. code-block:: bash

 dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \
   -f 'feat$1gram$msgs$user_id$0_05' --smooth_feats dirichlet --smooth_mu 500
