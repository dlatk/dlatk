************
Data Engines
************

DLATK offers two types of data engines: MySQL and SQLite. 


MySQL
=====

MySQL is the default data engine. Setting are passed to the data engine through a configuration file. By default DLATK tries to read from ``~/.my.cnf``. You can pass a configuration file to DLATK with `:doc:`../fwinterface/fwflag_mysql_config_file`. For example, if your config file was located at ``/home/username/mysql/config.txt`` then you would add the following to your command:

.. code-block:: bash

	--mysql_config_file /home/username/mysql/config.txt

We recommand that all configuration files have at least user, password, and host defined. Here is an example of this basic configuration.

.. code-block:: bash

	[client]
	user=your_mysql_username
	password=your_mysql_password
	host=the_mysql_server_host



SQLite
======

