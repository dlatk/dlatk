.. _fwflag_multir:
========
--multir
========
Switch
======

--multir

Description
===========

Multiple regression of each feature on all outcomes at once: which features each outcome explains, holding the other outcomes fixed.

Argument and Default Value
==========================

None

Details
=======

For each feature, one ordinary least squares regression is run with the feature (its group_norm) as y and all of the :doc:`fwflag_outcomes`, plus any :doc:`fwflag_outcome_controls`, together as X. Each outcome's coefficient is therefore its link to the feature after adjusting for the other outcomes and the controls. This differs from :doc:`fwflag_correlate`, which runs one outcome at a time. Only groups that have every outcome and control are used, so all outcomes share one set of groups.

As with :doc:`fwflag_correlate`:

* every variable is z-scored, so coefficients are standardized; :doc:`fwflag_spearman` ranks the variables first
* p-values are corrected across features, separately for each outcome (Benjamini-Hochberg by default; see :doc:`fwflag_p_correction`)
* controls adjust the model but their coefficients are not reported

:doc:`fwflag_interactions` also adds a product term for every pair of outcomes, reported as *outcome1##outcome2*: is the feature linked to the combination of outcomes beyond each one alone?

Results are written as the same matrix :doc:`fwflag_rmatrix` makes, to *OUTPUT_NAME_multir.html*: one row per feature and, for each outcome (and interaction), its standardized coefficient (*beta*), p-value, N (number of groups), confidence interval and the feature's frequency. :doc:`fwflag_csv` also writes *OUTPUT_NAME_multir.csv*, and :doc:`fwflag_sort` adds the features sorted by coefficient. Adding :doc:`fwflag_rmatrix` to the same command still gives the regular one-outcome-at-a-time correlations, in their own files.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t`, :doc:`fwflag_c`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes`

Optional Switches:

* :doc:`fwflag_outcome_controls`
* :doc:`fwflag_interactions`
* :doc:`fwflag_csv`, :doc:`fwflag_sort`, :doc:`fwflag_output_name`
* :doc:`fwflag_spearman`, :doc:`fwflag_p_correction`
* :doc:`fwflag_group_freq_thresh`

Example Commands
================
.. code-block:: bash

 # topics linked to age holding gender fixed (and vice versa), plus the age x gender interaction, adjusting for is_student
 dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \
   -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$1gra' \
   --outcome_table blog_outcomes --outcomes age gender --controls is_student \
   --multir --interactions --csv --sort --output_name multir_out
