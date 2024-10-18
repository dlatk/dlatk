.. _fwflag_logistic_reg:
==============
--logistic_reg
==============
Switch
======

--logistic_reg

Description
===========

Use logistic regression instead of linear regression. This is better for binary outcomes.

Argument and Default Value
==========================

None

Details
=======

Note: you cannot compare coefficients in Logistic regression. See this article for more info. You can compare p values, though. See :doc:`fwflag_correlate` for more info on correlation.


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t`, :doc:`fwflag_c`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes`

Optional Switches:

* :doc:`fwflag_cohens_d`
* :doc:`fwflag_interaction_ddla`
* :doc:`fwflag_correlate`
* :doc:`fwflag_rmatrix`
* :doc:`fwflag_tagcloud`
* :doc:`fwflag_topic_tagcloud`
* :doc:`fwflag_make_wordclouds`
* :doc:`fwflag_make_topic_wordclouds`

Example Commands
================

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id \ 
	-f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$1gra' \ 
	--outcome_table blog_outcomes  --group_freq_thresh 500 \ 
	--outcomes gender --output_name gender_correlates_logistic \ 
	--topic_tagcloud --make_topic_wordcloud --topic_lexicon met_a30_2000_freq_t50ll \ 
	--tagcloud_colorscheme bluered \
	--logistic_reg
