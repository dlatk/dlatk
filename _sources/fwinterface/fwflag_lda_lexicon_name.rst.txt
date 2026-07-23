.. _fwflag_lda_lexicon_name:
=================
--lda_lexicon_name
=================
Switch
======

--lda_lexicon_name [name]

Description
===========

Specifies the name of the LDA topic-lexicon to be created. Required unless :doc:`fwflag_no_lda_lexicon` is used.

Argument and Default Value
==========================

The name to use for storing the topic lexicon. Will be appended with `_cp` and `_freq_t50ll` for the two created tables.

Details
=======

See the :doc:`../tutorials/tut_lda` tutorial for details.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`
* :doc:`fwflag_estimate_lda_topics`

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
