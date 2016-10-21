.. _fwflag_where:
=======
--where
=======
Switch
======

--where VALUE

Description
===========

Filter groups based on sql call.

Argument and Default Value
==========================

Argument is sql-like query with column_name, operator and value. The column_name must be an actual column in --outcome_table. Do NOT include the word "where" after inside your. There is no default value.

Details
=======Contents [hide]Switch
Description
    Argument and Default Value
    
Other Switches
==============
    
Example Commands
================
.. code:doc:`fwflag_block`:: python
Switch

:doc:`fwflag_where` VALUE

Description

Filter groups based on sql call.

Argument and Default Value

Argument is sql:doc:`fwflag_like` query with column_name, operator and value. The column_name must be an actual column in :doc:`fwflag_outcome_table`. Do NOT include the word "where" after inside your. There is no default value.

Works with correlation, prediction and classification.

Examples:
:doc:`fwflag_where` "age >= 30 and age < 80"
:doc:`fwflag_where` "gender = 0"
:doc:`fwflag_where` "upt > 100"

Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` :doc:`fwflag_outcome_table` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Run LIWC over age and only consider males
 ./fwInterface.py :doc:`fwflag_d` dla_tutorial :doc:`fwflag_t` msgs_xxx :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$cat_LIWC2007$msgs_xxx$user_id$16to16' :doc:`fwflag_outcome_table` masterstats_r500  \ 
 :doc:`fwflag_outcomes` demog_age :doc:`fwflag_where` "demog_gender = 1" :doc:`fwflag_output_name` xxx_output