#!/bin/bash

# DLATK post-installation instructions.
# This script is executed when dlatkInterface.py is called with --colabify flag.

if ! mysql --version
then 
  echo "MySQL not found. Installing it now..."
  apt-get install mysql-server mysql-client -y
  echo '[mysqld]
  skip_log_bin=1
  ' >> /etc/mysql/mysql.conf.d/mysqld.cnf
  service mysql start
fi

DLATK_PATH=$1 #Path to DLATK passed from dlatkInterface.py
mysql < ${DLATK_PATH}/data/dla_tutorial.sql
mysql < ${DLATK_PATH}/data/dlatk_lexica.sql

# Install mallet 2.0.8
wget -q -O mallet.tar.gz http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz
mkdir -p /opt/mallet
tar -xf mallet.tar.gz -C /opt/mallet --strip-components=1
rm mallet.tar.gz

# Install Python dependencies
pip install -r /content/dlatk/install/requirements.txt
pip install rpy2==3.5.1
