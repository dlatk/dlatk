.. _tut_docker:
============================
Installing DLATK with Docker
============================

Run from Docker Hub
====================

Step 1: Install Docker
----------------------

Installing Docker is very easy. Visit the `official Docker installation page <https://docs.docker.com/install/>`_ page and follow the instructions tailored for your operating system. 

After you’ve installed Docker, open the terminal and type the following to verify the installation:

.. code-block:: bash

	> docker info


you should see something like 

.. code-block:: bash

	> docker info
	Containers: 0
	 Running: 0
	 Paused: 0
	 Stopped: 0
	Images: 0
	...



Step 2: Install MySQL
---------------------

You can pull the offical image of MySQL from `Docker Hub <https://hub.docker.com/_/mysql/>`_. Starting a MySQL instance is simple:

.. code-block:: bash

   > docker run --name some-mysql --env MYSQL_ROOT_PASSWORD=my-secret-pw --detach mysql:tag

where `some-mysql` is the name you want to assign to your container, `my-secret-pw` is the password to be set for the MySQL root user and tag is the `tag` specifying the MySQL version you want. We've tested using MySQL v5.5:

.. code-block:: bash

   > docker run --name mysql_v5  --env MYSQL_ROOT_PASSWORD=my-secret-pw --detach mysql:5.5

and we can confirm the installation with:

.. code-block:: bash
	
	> docker images
	REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
	mysql               5.5                 a8a59477268d        7 weeks ago         445MB

	> docker ps
	CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                    NAMES
	552954e73844        mysql               "docker-entrypoint..."   5 minutes ago       Up 5 minutes        0.0.0.0:3306->3306/tcp   mysql_v5

Note that this is running on port `3306`. Here is the command broken down:

* **run**: Run a command in a new container.
* **--name**: Assign a name to the container. If you don’t specify this, Docker will generate a random name.
* **--env**: Set environment variables
* **--detach**: Run container in background and print container ID
* **mysql**: The image name as stated on the Docker Hub page. This is the simplest image name. The standard is “username/image_name:tag”, for example “severalnines/mysql:5.6”. In this case, we specified “mysql”, which means it has no username (the image is built and maintained by Docker, therefore no username), the image name is “mysql” and the tag is latest (default). If the image does not exist, it will pull it first from Docker Hub into the host, and then run the container.

You can see which IP the MySQL container is running on via:

.. code-block:: bash
	
	> docker inspect mysql_v5 | grep IPAddress
            "SecondaryIPAddresses": null,
            "IPAddress": "172.17.0.2",
                    "IPAddress": "172.17.0.2",

Both of these can be used to configure a graphical SQL client such as Heidi, MySQL Workbench or Sequel Pro. 

To open this MySQL instance we run, remembering that we set the root password to `my-secret-pw`:

.. code-block:: bash
	
	> docker exec -it mysql_v5 bash

	root@d6ed6aa86c31:/# mysql -p
	Enter password:

	Welcome to the MySQL monitor.  Commands end with ; or \g.
	Your MySQL connection id is 13
	Server version: 8.0.11 MySQL Community Server - GPL

	Copyright (c) 2000, 2018, Oracle and/or its affiliates. All rights reserved.

	Oracle is a registered trademark of Oracle Corporation and/or its
	affiliates. Other names may be trademarks of their respective
	owners.

	Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

	mysql> show databases;
	+--------------------+
	| Database           |
	+--------------------+
	| information_schema |
	| mysql              |
	| performance_schema |
	| sys                |
	+--------------------+
	4 rows in set (0.00 sec)

	mysql> 

Step 3: Link MySQL and DLATK
----------------------------

Here we run DLATK and link to MySQL. We pull DLATK from it's `official repo at DockerHub <https://hub.docker.com/r/dlatk/dlatk/>`_:

.. code-block:: bash
	
	> docker run -it --rm --name dlatk_docker --link mysql_v5:mysql dlatk/dlatk bash

which should give you a new prompt. Here we can open MySQL as follows:

.. code-block:: bash
	
	root@70032e45f971:/# mysql -p
	Enter password: 
	Welcome to the MariaDB monitor.  Commands end with ; or \g.
	Your MySQL connection id is 1
	Server version: 5.5.60 MySQL Community Server (GPL)

	Copyright (c) 2000, 2017, Oracle, MariaDB Corporation Ab and others.

	Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

	MySQL [(none)]> show databases;
	+--------------------+
	| Database           |
	+--------------------+
	| information_schema |
	| mysql              |
	| performance_schema |
	+--------------------+
	3 rows in set (0.00 sec)

	MySQL [(none)]> exit
	Bye


Next we will upload the sample data packaged with DLATK into MySQL, noting that we have access to the DLATK install path via `$DLATK_DIR`:

.. code-block:: bash
	
	root@70032e45f971:/# echo $DLATK_DIR
	/usr/local/lib/python3.6/site-packages/dlatk

	root@70032e45f971:/# mysql < $DLATK_DIR/data/dla_tutorial.sql
	root@70032e45f971:/# mysql < $DLATK_DIR/data/permaLexicon.sql
	root@70032e45f971:/# mysql
	Welcome to the MariaDB monitor.  Commands end with ; or \g.
	Your MySQL connection id is 4
	Server version: 5.5.60 MySQL Community Server (GPL)

	Copyright (c) 2000, 2017, Oracle, MariaDB Corporation Ab and others.

	Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

	MySQL [(none)]> show databases;
	+--------------------+
	| Database           |
	+--------------------+
	| information_schema |
	| dla_tutorial       |
	| mysql              |
	| performance_schema |
	| permaLexicon       |
	+--------------------+
	5 rows in set (0.00 sec)

Going back to the prompt we can run DLATK through the interface script `dlatkInterface.py`:

.. code-block:: bash
	
	root@70032e45f971:/# dlatkInterface.py -h

Note that this also installs Mallet, Stanford Parser and Tweet NLP with Mallet added to your path:

.. code-block:: bash
	
	root@0f8f18074713:/# mallet
	Unrecognized command: 
	Mallet 2.0 commands: 

	  import-dir         load the contents of a directory into mallet instances (one per file)
	  import-file        load a single file into mallet instances (one per line)
	  import-svmlight    load SVMLight format data files into Mallet instances
	  info               get information about Mallet instances
	  train-classifier   train a classifier from Mallet data files
	  classify-dir       classify data from a single file with a saved classifier
	  classify-file      classify the contents of a directory with a saved classifier
	  classify-svmlight  classify data from a single file in SVMLight format
	  train-topics       train a topic model from Mallet data files
	  infer-topics       use a trained topic model to infer topics for new documents
	  evaluate-topics    estimate the probability of new documents under a trained model
	  prune              remove features based on frequency or information gain
	  split              divide data into testing, training, and validation portions
	  bulk-load          for big input files, efficiently prune vocabulary and import docs

	Include --help with any option for more information


Build Image from DockerFile
===========================

This is more advanced and probably not needed for most use cases. First we download the DockerFile from GitHub. If you have `git` installed you can run 

.. code-block:: bash

	> git clone https://github.com/dlatk/dlatk-docker.git && cd dlatk-docker
   
To build the image we run:

.. code-block:: bash
	
	> docker build -t dlatk-docker .

Here is the command broken down:

* **build**: Build an image from a Dockerfile
* **-t**: Alias for `--tag`. Name and optionally a tag in the 'name:tag' format. Since we are not specifying a tag we will pull the latest version.

You will see the following output:

.. code-block:: bash
	
	Sending build context to Docker daemon  84.48kB
	Step 1/15 : FROM python:3.6-stretch
	stretch: Pulling from library/python
	cc1a78bfd46b: Downloading [=============================================>     ]  40.86MB/45.32MB
	d2c05365ee2a: Download complete 
	231cb0e216d3: Download complete 
	3d2aa70286b8: Downloading [===================================>               ]  35.08MB/50.06MB
	e80dfb6a4adf: Downloading [=======>                                           ]  31.16MB/213.2MB
	....

At the end you should see:

.. code-block:: bash
	
	Removing intermediate container c4776548e966
	Successfully built dc2005cd24a6
	Successfully tagged dlatk-docker:latest

and we can confirm the installation with:

.. code-block:: bash
	
	> docker images
	docker images
	REPOSITORY          TAG                 IMAGE ID            CREATED              SIZE
	dlatk-docker        latest              10eea3e0202a        About a minute ago   2.56GB
	python              3.6-stretch         d330010a503a        3 days ago           912MB

Acknowledgment
==============

The DockerFile was originally written by `Michael Becker <https://github.com/mdbecker>`_ at Penn Medicine. 

