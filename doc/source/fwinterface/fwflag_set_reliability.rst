.. _fwflag_set_reliability:
=================
--set_reliability
=================
Switch
======

--set_reliability R

Description
===========

The minimum split-half reliability a feature needs to be kept by :doc:`fwflag_feat_reliability_filter`.

Argument and Default Value
==========================

A number below 1; default 0.30.

Details
=======

0.30 keeps features whose relative frequency across groups is at least 30% signal; a feature's correlations with outcomes are then attenuated by at most a factor of sqrt(0.30) = 0.55. Most n-grams are less reliable than that at a few thousand words per group (e.g. users of a blog corpus), so 0.30 already removes most rare n-grams; higher values keep fewer, more reliable features, and groups with more words make more features reliable. The new table is named by the threshold (e.g. *$rel0_3*, or *$rel0_6* with --set_reliability 0.6).

Other Switches
==============

Required Switches:

* :doc:`fwflag_feat_reliability_filter`

Example Commands
================
.. code-block:: bash

 dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \
   -f 'feat$1to3gram$msgs$user_id$0_05' --feat_reliability_filter --set_reliability 0.6
