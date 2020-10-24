.. _fwflag_num_stopwords:
=================
--num_stopwords
=================
Switch
======

--num_stopwords [num]

Description
===========

Set the number of stop words to filter out of LDA topic estimation. Stop words are automatically set to be the top `[num]` most frequent words in the unigrams table.

Argument and Default Value
==========================

The number of stop words to use. Default: 50.


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
* :doc:`../fwinterface/fwflag_no_lda_stopping`
* :doc:`../fwinterface/fwflag_lda_alpha`
* :doc:`../fwinterface/fwflag_lda_beta`
* :doc:`../fwinterface/fwflag_lda_iterations`


Example Commands
================

See the :doc:`../tutorials/tut_lda` tutorial for examples.
