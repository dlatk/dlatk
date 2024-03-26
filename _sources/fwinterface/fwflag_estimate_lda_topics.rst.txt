.. _fwflag_estimate_lda_topics:
=================
--estimate_lda_topics
=================
Switch
======

--estimate_lda_topics

Description
===========

Starts the automated LDA topic estimation process using either PyMallet or Mallet.

Argument and Default Value
==========================


Details
=======

Will run topic estimation using PyMallet unless :doc:`fwflag_mallet_path` is specified.

See the :doc:`../tutorials/tut_lda` tutorial for details.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`
* :doc:`fwflag_lda_lexicon_name` or :doc:`fwflag_no_lda_lexicon`

Optional Switches:

* :doc:`../fwinterface/fwflag_mallet_path`
* :doc:`../fwinterface/fwflag_save_lda_files`
* :doc:`../fwinterface/fwflag_num_topics`
* :doc:`../fwinterface/fwflag_num_stopwords`
* :doc:`../fwinterface/fwflag_no_lda_stopping`
* :doc:`../fwinterface/fwflag_lda_alpha`
* :doc:`../fwinterface/fwflag_lda_beta`
* :doc:`../fwinterface/fwflag_lda_iterations`


Example Commands
================

See the :doc:`../tutorials/tut_lda` tutorial for examples.
