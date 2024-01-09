#!/bin/bash
# DLATK post-installation instructions.
# This script is executed when dlatkInterface.py is called with --colabify flag.

DLATK_PATH=$1 #Path to DLATK passed from dlatkInterface.py

mkdir -p /content/sqlite_data
cp ${DLATK_PATH}/data/dlatk_lexica.db /content/sqlite_data/dlatk_lexica.db
cp ${DLATK_PATH}/data/msgs* /content/
cp ${DLATK_PATH}/data/users* /content/

# Install mallet 2.0.8
wget -q -O mallet.tar.gz http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz
mkdir -p /opt/mallet
tar -xf mallet.tar.gz -C /opt/mallet --strip-components=1
rm mallet.tar.gz
