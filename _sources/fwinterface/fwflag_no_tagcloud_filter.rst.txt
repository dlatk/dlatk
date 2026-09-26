.. _fwflag_no_tagcloud_filter:
====================
--no_tagcloud_filter
====================
Switch
======

--no_tagcloud_filter

Description
===========

If given, turns OFF duplicate filtering of tagclouds.

Argument and Default Value
==========================

The hardcoded threshold in featureWorker is 0.25; i.e., if a cloud shares more than 25% of its terms with a previously-constructed one for a given outcome, it is marked as duplicate in the tagcloud text file/cloud image filename.

Details
=======


Other Switches
==============

Required Switches:
:doc:`fwflag_tagcloud` OR
:doc:`fwflag_topic_tagcloud` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


See tagcloud or topic_tagcloud.

