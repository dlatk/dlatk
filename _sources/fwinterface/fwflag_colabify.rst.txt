.. _fwflag_colabify:
===========
--colabify
===========
Switch
======

--colabify

Description
===========

Flag that facilitates DLATK support to Google Colab by running a post-installation script.

Argument and Default Value
==========================

None

Details
=======

This flag is required when DLATK is used in Google Colab. When used, dlatkInterface.py checks if it's a Colab environment and calls a post-installation script at {DLATK_PATH}/colabify.sh. However, it's important to use it before any further DLATK commands, and also without any other flags.

Currently, it installs - 
* Python 3.6
* MySQL 5.7 (imports `dlatk_lexica` and `dla_tutorial`)
* Mallet

Other Switches
==============

Required Switches:

None

Example Commands
================

.. code-block:: bash


	dlatkInterface.py --colabify
