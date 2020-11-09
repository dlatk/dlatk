.. _fwflag_save_lda_files:
=================
--save_lda_files
=================
Switch
======

--save_lda_files [/path/to/save]

Description
===========

Specifies the location in which interim LDA estimation files should be saved. Useful if you want to view the estimation process step-by-step.

Argument and Default Value
==========================

Specify the directory to save files in. ``/tmp`` is used by default.

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
