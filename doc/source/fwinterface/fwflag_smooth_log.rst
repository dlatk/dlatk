.. _fwflag_smooth_log:
============
--smooth_log
============
Switch
======

--smooth_log

Description
===========

With :doc:`fwflag_smooth_feats` dirichlet, nb, pln or zinb, stores each feature's posterior expected log rate instead of its rate.

Argument and Default Value
==========================

None

Details
=======

Across users and counties, the rates of most words, topics and lexica are close to log-normal (Almodaresi et al., 2017), so their logs are close to Normal, which suits correlations and linear models. A log needs a nonzero rate, which smoothing provides. The value is the expected log rate given the group's count and total (not the log of the smoothed rate), which is more accurate for small counts. The table name gets an *L* (e.g. *$dirL*, *$nbL*). For dirichlet, every group uses the corpus-wide *mu* (see :doc:`fwflag_smooth_feats`).

Other Switches
==============

Required Switches:

* :doc:`fwflag_smooth_feats` dirichlet, nb, pln or zinb

Optional Switches:

* :doc:`fwflag_smooth_zero_gap`

Alternative: :doc:`fwflag_smooth_sqrt`

Example Commands
================
.. code-block:: bash

 dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \
   -f 'feat$1gram$msgs$user_id$0_05' --smooth_feats pln --smooth_log
