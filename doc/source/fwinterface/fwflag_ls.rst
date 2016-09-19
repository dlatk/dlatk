.. _fwflag_ls:
====
--ls
====
Switch
======

--ls

Description
===========

See all available feature tables given database, message table and correl field.

Argument and Default Value
==========================

Required Switches:

Details
=======Contents [hide]Switch
Description
    
Other Switches
==============
    
Example Commands
================
.. code:doc:`fwflag_block`:: python
Switch

:doc:`fwflag_ls` 

Description

See all available feature tables given database, message table and correl field.


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 ./fwInterface.py :doc:`fwflag_d` dla_tutorial :doc:`fwflag_t` msgs_xxx :doc:`fwflag_c` user_id :doc:`fwflag_ls` OUTPUT
 SQL QUERY: SHOW TABLES FROM dla_tutorial LIKE 'feat$%$msgs_xxx$user_id$%' 
 Found 18 available feature tables
 feat$1gram$msgs_xxx$user_id$16to16
 feat$1gram$msgs_xxx$user_id$16to16$0_001
 feat$1gram$msgs_xxx$user_id$16to16$0_01
 feat$1gram$msgs_xxx$user_id$16to16$0_1
 feat$1gram$msgs_xxx$user_id$16to4
 feat$1to3gram$msgs_xxx$user_id$16to16
 feat$1to3gram$msgs_xxx$user_id$16to16$0_001
 feat$1to3gram$msgs_xxx$user_id$16to16$0_05
 feat$2gram$msgs_xxx$user_id$16to16
 feat$3gram$msgs_xxx$user_id$16to16
 feat$cat_LIWC2007$msgs_xxx$user_id$16to16
 feat$cat_dd_emnlp14_ageGender_w$msgs_xxx$user_id$16to16
 feat$cat_met_a30_2000_cp_w$msgs_xxx$user_id$16to16
 feat$cat_w2v_200_w$msgs_xxx$user_id$16to16
 feat$meta_1gram$msgs_xxx$user_id$16to16
 feat$meta_1gram$msgs_xxx$user_id$16to4
 feat$meta_2gram$msgs_xxx$user_id$16to16
 feat$meta_3gram$msgs_xxx$user_id$16to16