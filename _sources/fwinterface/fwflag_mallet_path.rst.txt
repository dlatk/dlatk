.. _fwflag_mallet_path:
=================
--mallet_path
=================
Switch
======

--mallet_path /path/to/mallet/bin

Description
===========

Specifies the Mallet installation bin directory, enabling the LDA interface to estimate topics using Mallet (instead of PyMallet).

Argument and Default Value
==========================

/path/to/mallet/bin must be specified. There is no default.


Details
=======

See the :doc:`../tutorials/tut_lda` tutorial for details.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`
* :doc:`../fwinterface/fwflag_estimate_lda_topics`
* :doc:`fwflag_lda_lexicon_name` or :doc:`fwflag_no_lda_lexicon`

Optional Switches:

* :doc:`../fwinterface/fwflag_save_lda_files`
* :doc:`../fwinterface/fwflag_lda_lexicon_name`
* :doc:`../fwinterface/fwflag_no_lda_lexicon`
* :doc:`../fwinterface/fwflag_num_topics`
* :doc:`../fwinterface/fwflag_num_stopwords`
* :doc:`../fwinterface/fwflag_no_lda_stopping`
* :doc:`../fwinterface/fwflag_lda_alpha`
* :doc:`../fwinterface/fwflag_lda_beta`
* :doc:`../fwinterface/fwflag_lda_iterations`


Example Commands
================

See the :doc:`../tutorials/tut_lda` tutorial for examples.
