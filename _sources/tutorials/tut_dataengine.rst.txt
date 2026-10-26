************
Data Engines
************

DLATK offers two types of data engines: MySQL and SQLite. 


MySQL
=====

MySQL is the default data engine. Settings are passed to the data engine through a configuration file. By default DLATK tries to read from ``~/.my.cnf``. You can pass a configuration file to DLATK with :doc:`../fwinterface/fwflag_mysql_config_file`. For example, if your config file was located at ``/home/your_username/mysql/config.txt`` then you would add the following to your command:

.. code-block:: bash

	dlatkInterface.py --mysql_config_file /home/your_username/mysql/config.txt 

We recommand that all configuration files have at least user, password, and host defined. Here is an example of this basic configuration:

.. code-block:: none

	[client]
	user=your_mysql_username
	password=your_mysql_password
	host=the_mysql_server_host

Python MySQL dependecies are not automatically installed with DLATK. To install them you can use pip or conda. For example, 

.. code-block:: bash

	pip install mysqlclient
	pip install SQLAlchemy

The following versions are known to work with DLATK v1.2.0 in Python 3.8:

.. code-block:: bash

	pip install 'mysqlclient==2.0.1'
	pip install 'SQLAlchemy==1.3.20'

Note, that if the file ``~/.my.cnf`` does not exist then MySQL will use the following defaults:

* User: your current system-level username
* Password: No password
* Host: ``localhost``

SQLite
======

The data packaged with DLATK is formatted for MySQL. To use SQLite you must first convert these MySQL dumps to a SQLite database. You can do this with the `mysql2sqlite <https://github.com/dumblob/mysql2sqlite>`_ package. Clone this package with

.. code-block:: bash

	git clone https://github.com/dumblob/mysql2sqlite.git


Next, we convert the MySQL dumps to SQLite databases:

.. code-block:: bash

	cd mysql2sqlite
	./mysql2sqlite /path/to/dlatk/data/dla_tutorial.sql | sqlite3 /path/to/dlatk/data/dla_tutorial.db
	./mysql2sqlite /path/to/dlatk/data/dlatk_lexica.sql | sqlite3 /path/to/dlatk/data/dlatk_lexica.db

In order switch from the default MySQL to SQLite you need to add the following flag to your commands :doc:`../fwinterface/fwflag_db_engine` and use the full path of the SQLite db file. For example,

.. code-block:: bash

	dlatkInterface.py --db_engine sqlite -d /path/to/dlatk/data/dla_tutorial  


The Python dependency for SQLite (``sqlite3``) is part of the standard library, so no additional packages are necessary. 
