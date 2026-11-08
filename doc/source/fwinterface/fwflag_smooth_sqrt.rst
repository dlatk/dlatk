.. _fwflag_smooth_sqrt:
=============
--smooth_sqrt
=============
Switch
======

--smooth_sqrt

Description
===========

With :doc:`fwflag_smooth_feats` dirichlet, nb, pln or zinb, stores the square root of each feature's smoothed rate instead of the rate.

Argument and Default Value
==========================

None

Details
=======

The square root compresses differences among large rates less than a log does, and is what the unsmoothed *--sqrt* option (16to4 tables) does to relative frequencies, which often predicts well. The table name gets an *S* (e.g. *$nbS*). It is an alternative to :doc:`fwflag_smooth_log`; use one or the other. With :doc:`fwflag_smooth_zero_gap`, groups that never used a feature stay at 0.

Other Switches
==============

Required Switches:

* :doc:`fwflag_smooth_feats` dirichlet, nb, pln or zinb

Optional Switches:

* :doc:`fwflag_smooth_zero_gap`

Example Commands
================
.. code-block:: bash

 dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \
   -f 'feat$1gram$msgs$user_id$0_05' --smooth_feats nb --smooth_sqrt
 # creates feat$1gram$msgs$user_id$0_05$nbS
