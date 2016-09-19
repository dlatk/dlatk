.. _fwflag_no_metafeats:
==============
--no_metafeats
==============
Switch
======

--no_metafeats

Description
===========

If given, do not extract meta features (word, message length) with ngrams.

Argument and Default Value
==========================

None

Details
=======


Other Switches
==============

Required Switches:
:doc:`fwflag_add_ngrams` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


./fwInterface.py :doc:`fwflag_d` 2004blogs :doc:`fwflag_t` messages :doc:`fwflag_c` user_id :doc:`fwflag_add_ngrams` :doc:`fwflag_no_metafeats` 
Without :doc:`fwflag_no_metafeats`, this would create both feat$1gram$messages$user_id$16to16 (the 1:doc:`fwflag_gram` table) and feat$meta_1gram$messages$user_id$16to16. :doc:`fwflag_no_metafeats` turns off creation of the latter.

