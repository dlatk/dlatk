.. _fwflag_reduced_lexicon:
=================
--reduced_lexicon
=================
Switch
======

--reduced_lexicon table_name

Description
===========

Refer to a reduced lexicon table previously created when creating super topics. 

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

Optional Switches:

* :doc:`../fwinterface/fwflag_super_topics`

Example Commands
================

First we create the reduced lexicon table with: 

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' --fit_reducer --model nmf --reducer_to_lexicon msgs_reduced10_nmf --n_components 10

Then we can refer to the above table during downstream analysis:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id  --reduced_lexicon msgs_reduced10_nmf --super_topics msgs_10nmf_fbcp -l met_a30_2000_cp

