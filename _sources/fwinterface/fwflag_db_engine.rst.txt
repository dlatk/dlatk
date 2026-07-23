.. _fwflag_db_engine:
===========
--db_engine
===========
Switch
======

--db_engine name_of_engine

Description
===========

Set the backend engine (either sqlite or mysql). Can also use -e or --engine

Argument and Default Value
==========================

Default value: mysql

Details
=======

For more details on the data engines please see :doc:`../tutorials/tut_dataengine`.


Example Commands
================

Standard unigram extraction using sqlite

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_ngrams --db_engine sqlite




