.. _fwflag_super_topics:
==============
--super_topics
==============
Switch
======

--super_topics table_name

Description
===========

Unroll reduced topics to the word level. The lexicon MySQL table will be written to the *permaLexicon* database by default.

Argument and Default Value
==========================

MySQL table name. There is no default.

Details
=======

See the :doc:`../tutorials/tut_clustering` tutorial for details.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 
* :doc:`../fwinterface/fwflag_fit_reducer`
* :doc:`../fwinterface/fwflag_model`
* :doc:`../fwinterface/fwflag_reducer_to_lexicon` or :doc:`../fwinterface/fwflag_reduced_lexicon`
* :doc:`../fwinterface/fwflag_l`


Example Commands
================

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id  --reduced_lexicon msgs_reduced10_nmf --super_topics msgs_10nmf_fbcp -l met_a30_2000_cp



