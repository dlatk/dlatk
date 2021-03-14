.. _fwflag_mysql_config_file:
===================
--mysql_config_file
===================
Switch
======

--mysql_config_file /path/to/config/file.txt

Description
===========

Set the location of the MySQL configuration file. Can also use --conf or --mysql_config

Argument and Default Value
==========================

Default value: ~/.my.cnf

Details
=======

For more details on the data engines please see :doc:`../tutorials/tut_dataengine`.


Example Commands
================

Standard unigram extraction

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_ngrams --mysql_config_file ~/mysql_config.txt




