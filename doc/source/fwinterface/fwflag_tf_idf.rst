.. _fwflag_tf_idf:
========
--tf_idf
========
Switch
======

--tf_idf

Description
===========

Creates new feature table where the group_norm is the tf-idf score. Each group_id is seen as a document for calculating tf-idf.

Argument and Default Value
==========================

None

Details
=======

:doc:`fwflag_f` should be an ngram table.

Resulting value refers to value in ngram table. Group_norm refers to tf:doc:`fwflag_idf` score.


Other Switches
==============

Required: :doc:`fwflag_d`, :doc:`fwflag_t`, :doc:`fwflag_f` 

Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Creates tf:doc:`fwflag_idf` table at plu.feat$tf_idf_1gram$msgsEn$user_id 
 ~/fwInterface.py :doc:`fwflag_d` plu :doc:`fwflag_t` msgsEn :doc:`fwflag_f` 'feat$1gram$msgsEn$user_id$16to16' :doc:`fwflag_tf_idf` 