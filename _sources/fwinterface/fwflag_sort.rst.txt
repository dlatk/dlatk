.. _fwflag_sort:
======
--sort
======
Switch
======

--sort

Description
===========

Modifies the output of a correlation that is generated based on options such as --correlate, --rmatrix and --csv.  Usually it modifies the output to appear once unsorted, and again sorted.

Argument and Default Value
==========================

null

Details
=======Contents [hide]Switch
Description
    
Example Commands
================
.. code:doc:`fwflag_block`:: python
Switch

:doc:`fwflag_sort` 
Description

Modifies the output of a correlation that is generated based on options such as :doc:`fwflag_correlate`, :doc:`fwflag_rmatrix` and :doc:`fwflag_csv`.  Usually it modifies the output to appear once unsorted, and again sorted.  


Example Commands
================
.. code:doc:`fwflag_block`:: python


 ./fwInterface.py :doc:`fwflag_H` wwbp :doc:`fwflag_d` testing :doc:`fwflag_t` msgs :doc:`fwflag_c` user_id :doc:`fwflag_f` feat$1to3gram$msgs$user_id$16to16 :doc:`fwflag_outcome_table` users :doc:`fwflag_outcomes` age salary \ 
 :doc:`fwflag_output_name` test_correlation :doc:`fwflag_rmatrix` :doc:`fwflag_csv` :doc:`fwflag_sort` 