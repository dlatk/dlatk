.. _fwflag_mediation_no_summary:
======================
--mediation_no_summary
======================
Switch
======

--mediation_no_summary

Description
===========

Mediation analysis, by default, prints a summary report of all significant results (using the Sobel p value). This switch turns that off.

Argument and Default Value
==========================

Required Switches:

Details
=======

Mediation analysis, by default, prints a summary report of all significant results (using the Sobel p value). This switch turns that off.


Other Switches
==============

Required Switches:
:doc:`fwflag_mediation` Note: there are many other required and optional switches when running :doc:`fwflag_mediation`. 

Example Commands
================
.. code:doc:`fwflag_block`:: python


 ./fwInterface.py :doc:`fwflag_d` twitterGH :doc:`fwflag_t` messages_en :doc:`fwflag_c` cty_id :doc:`fwflag_f` 'feat$cat_met_a30_2000_cp_w$messages_en$cty_id$16to16' :doc:`fwflag_outcome_table` nejm_intersect_small50k \ 
 :doc:`fwflag_mediation` :doc:`fwflag_path_start` 'hsgradHC03_VC93ACS3yr$10'  :doc:`fwflag_outcomes` 'ucd_I25_1_atheroHD$0910_ageadj' :doc:`fwflag_mediators`  '1051' :doc:`fwflag_mediation_no_summary` :doc:`fwflag_mediation_csv` 