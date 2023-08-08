#!/bin/bash
# DLATK post-installation instructions.
# This script is executed when dlatkInterface.py is called with --colabify flag.

# Install mallet 2.0.8
wget -q -O mallet.tar.gz http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz
mkdir -p /opt/mallet
tar -xf mallet.tar.gz -C /opt/mallet --strip-components=1
rm mallet.tar.gz

# Install Python dependencies
pip install -r /content/dlatk/install/requirements.txt
pip install rpy2==3.5.1
