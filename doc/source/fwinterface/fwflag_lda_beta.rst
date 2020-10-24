.. _fwflag_lda_beta:
=================
--lda_beta
=================
Switch
======

--lda_beta [value]

Description
===========

Set the LDA beta hyperparameter, which is a prior on the number of words per topic. We typically do not change this hyperparameter.

Note that this command is currently ignored by Mallet; it does affect PyMallet.

Argument and Default Value
==========================

Value of the beta parameter. Default: 0.01.


Details
=======

See the :doc:`../tutorials/tut_lda` tutorial for details.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`
* :doc:`fwflag_estimate_lda_topics`
* :doc:`fwflag_lda_lexicon_name` or :doc:`fwflag_no_lda_lexicon`

Optional Switches:

* :doc:`../fwinterface/fwflag_mallet_path`
* :doc:`../fwinterface/fwflag_save_lda_files`
* :doc:`../fwinterface/fwflag_lda_lexicon_name`
* :doc:`../fwinterface/fwflag_no_lda_lexicon`
* :doc:`../fwinterface/fwflag_num_topics`
* :doc:`../fwinterface/fwflag_num_stopwords`
* :doc:`../fwinterface/fwflag_no_lda_stopping`
* :doc:`../fwinterface/fwflag_lda_alpha`
* :doc:`../fwinterface/fwflag_lda_iterations`


Example Commands
================

See the :doc:`../tutorials/tut_lda` tutorial for examples.
