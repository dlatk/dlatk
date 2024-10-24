.. _fwflag_csv:
=====
--csv
=====
Switch
======

--csv

Description
===========

Generates a csv output when using the :doc:`fwflag_correlate` switch.

Other Switches
==============

Optional Switches:

* :doc:`fwflag_sort` 

Example Commands
================

.. code-block:: bash


	dlatkInterface.py -d testing -t msgs -c user_id -f 'feat$1to3gram$msgs$user_id$16to16' --outcome_table users --outcomes age salary --output_name test_correlation --rmatrix --csv --sort 