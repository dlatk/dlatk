.. _fwflag_corp_topic_tagcloud:
=====================
--corp_topic_tagcloud
=====================
Switch
======

--corp_topic_tagcloud

Description
===========

Produces data for making topic Wordles. You must use a topic based feature table. For a NON topic based word cloud see :doc:`fwflag_tagcloud`.

Other Switches
==============

Required Switches:

* :doc:`fwflag_f` FEAT_TABLE (must be topic based)
* :doc:`fwflag_topic_lexicon`
* :doc:`fwflag_outcome_table`
* :doc:`fwflag_outcomes` 

Optional Switches

* :doc:`fwflag_make_topic_wordclouds`
* :doc:`fwflag_tagcloud_filter` / :doc:`fwflag_no_tagcloud_filter` 

Example Commands
================

.. code-block:: bash
