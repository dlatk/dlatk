.. _tut_import_methods:
============================
Importing and Exporting Data
============================

DLATK is packged with tools to import and export data from MySQL. There are two options: a standalone Python script (similar to dlatkInterface.py) or you can use Python. 

Standalone Script
=================

CSV to MySQL
------------

If your MySQL table does not exist, then use the following to upload your CSV:

.. code-block:: bash

   ./dlatk/tools/importmethods.py -d database -t table --csv_to_mysql --csv_file /path/to/file.csv --column_description "some mysql column description" [--ignore_lines N]

If your table already exists then you can append the CSV file with

.. code-block:: bash

   ./dlatk/tools/importmethods.py -d database -t table --append_csv_to_mysql --csv_file /path/to/file.csv [--ignore_lines N]

MySQL to CSV
------------

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --mysql_to_csv --csv_file /path/to/file.csv



Using Python
============

You can also use Python to import/export your data.

CSV to MySQL
------------

.. code-block:: python
	
	import dlatk
	dlatk.csvToMySQL(csvFile, database, table, columnDescription, ignoreLines=0)



MySQL to CSV
------------

.. code-block:: python
	
	import dlatk
	dlatk.mySQLToCSV(database, table, csvFile)
