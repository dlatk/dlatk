.. _fwflag_reducer_to_lexicon:
====================
--reducer_to_lexicon
====================
Switch
======

--reducer_to_lexicon table_name

Description
===========

Writes the reduction model to a specified lexicon. The lexicon MySQL table will be written to the *dlatk_lexica* database by default.

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

Optional Switches:

* :doc:`../fwinterface/fwflag_super_topics`

Example Commands
================

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' --fit_reducer --model nmf --reducer_to_lexicon msgs_reduced10_nmf --n_components 10



