.. _fwflag_outcome_table:
===============
--outcome_table
===============
Switch
======

--outcome_table

Description
===========

Table containing the values for the outcomes.

Argument and Default Value
==========================

Name of MySQL table

Details
=======

This specifies the table name that contains the values of the :doc:`fwflag_outcomes` (and :doc:`fwflag_outcome_controls`, etc.). There should only be one row per :doc:`fwflag_c`. Should be in the working database specified under :doc:`fwflag_d`, and should have an :doc:`fwflag_c` index on the column.