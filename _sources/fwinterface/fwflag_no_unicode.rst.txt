.. _fwflag_no_unicode:
============
--no_unicode
============
Switch
======

--no_unicode

Description
===========

If set then all CHAR, VARCHAR and TEXT fields in MySQL will be treated as strings. This is done both in reading from MySQL and writing to MySQL (and writing to any other output formats specified: csv, html, wordclouds). When printing wordclouds with this flag all features with non-ascii will be ignored.

Argument and Default Value
==========================

If this flag is set during feature extraction all non-ascii will be removed.

Details
=======Contents [hide]Switch
Description
    Argument and Default Value
    
Example Commands
================
.. code:doc:`fwflag_block`:: python
Switch

:doc:`fwflag_no_unicode` 
Description

If set then all CHAR, VARCHAR and TEXT fields in MySQL will be treated as strings. This is done both in reading from MySQL and writing to MySQL (and writing to any other output formats specified: csv, html, wordclouds). When printing wordclouds with this flag all features with non:doc:`fwflag_ascii` will be ignored. 

If this flag is set during feature extraction all non:doc:`fwflag_ascii` will be removed.

Argument and Default Value

Unicode is the default and no flag is needed. Only use this flag for turning it off. 

This flag sets the encoding to latin1. You can override this with :doc:`fwflag_encoding`. 


Example Commands
================
.. code:doc:`fwflag_block`:: python


This command will write all output to HTML and CSV files as strings. 

 ./fwInterface.py :doc:`fwflag_d` dla_tutorial :doc:`fwflag_t` msgs_xxx :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$cat_LIWC2007$msgs_xxx$user_id$16to16' :doc:`fwflag_outcome_table` masterstats_r500 :doc:`fwflag_group_freq_thresh` 1000 \ 
 :doc:`fwflag_outcomes` demog_age demog_gender :doc:`fwflag_output_name` xxx_output :doc:`fwflag_rmatrix` :doc:`fwflag_sort` :doc:`fwflag_csv` :doc:`fwflag_no_unicode` 