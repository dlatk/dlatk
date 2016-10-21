.. _fwflag_feat_colloc_filter:
====================
--feat_colloc_filter
====================
Switch
======

--feat_colloc_filter

Description
===========

Filter multigram features based on how commonly they appear together

Argument and Default Value
==========================

This filters multigram features based on their PMI value/(number of words -1) and creates a new feature table that contains only the multi grams that were above a given threshold.  The PMI value for a bigram b composed of word1 followed by word2 is calculated as follows:

Details
=======

This filters multigram features based on their PMI value/(number of words :doc:`fwflag_1`) and creates a new feature table that contains only the multi grams that were above a given threshold.  The PMI value for a bigram b composed of word1 followed by word2 is calculated as follows:



In this case  is the number of times x shows up divided by the total number of words in a document.

Intuitively the PMI should be a measure of how much a word pair "goes together".
The PMI of a bigram like "happy birthday" will have a larger PMI.
The PMI of a bigram like "bird purple" will have a smaller PMI because it was probably just a random occurence.

We divide the PMI by the number of words minus 1 in order to normalize.. otherwise 3grams have much higher PMI values than 2grams.

For more information see the article:
Normalized (Pointwise) Mutual Information in Collocation Extraction by Gerlof Bouma


Other Switches
==============

Required Switches:
None, pmi threshold will be set to it's default which is 3.0
Optional Switches:
:doc:`fwflag_set_pmi_threshold` <val>

Example Commands
================
.. code:doc:`fwflag_block`:: python


./fwInterface.py :doc:`fwflag_d` fb22 :doc:`fwflag_t` messagesEn :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$2to3gram$messagesEn$user_id$16to16$0_02' :doc:`fwflag_feat_colloc_filter` :doc:`fwflag_set_pmi_threshold` 6.0

