#!/bin/bash

# DLATK post-installation instructions.
# This script is executed when dlatkInterface.py is called with --colabify flag.

#Install Python 3.6 if not installed already
if ! python --version | grep -q "3.6" 
then 
  echo "Python 3.6 not found. Installing it now..."
  update-alternatives --install /usr/local/bin/python python /usr/bin/python3.6 1
  update-alternatives --set python /usr/bin/python3.6
  python --version
fi

#Install MySQL 5.7 if not installed already
if ! mysql --version | grep -q "5.7" 
then 
  echo "MySQL 5.7 not found. Installing it now..."
  apt-get install mysql-server-5.7 mysql-client-5.7
  service mysql start
fi

DLATK_PATH=$1 #Path to DLATK passed from dlatkInterface.py
mysql < ${DLATK_PATH}/data/dla_tutorial.sql
mysql < ${DLATK_PATH}/data/dlatk_lexica.sql

#Install mallet 2.0.8
wget -q -O mallet.tar.gz http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz
mkdir -p /opt/mallet
tar -xf mallet.tar.gz -C /opt/mallet --strip-components=1
rm mallet.tar.gz
