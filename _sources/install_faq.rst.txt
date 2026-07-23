************
Install FAQs
************

1. Most install issues can be solved with updating pip

	``pip install --upgrade pip``

	or

	``pip3 install --upgrade pip``


2. Linux: ``Do you want to continue? [Y/n] Abort.``

	``sudo xargs apt-get -y install < install/requirements.sys``

	OR

	``sudo xargs apt-get -y --force-yes install < install/requirements.sys``

	WARNING: this will automatically install everything in the requirements.sys file.

3. mysqlclient: ``OSError: mysql_config not found``

	Make sure MySQL is installed and running before trying to install mysqlclient with pip.

4. ``ImportError: No module named 'numpy'``

	You must install numpy manually and then rerun ``pip install dlatk``:

	``pip install numpy``

5. Packaged datasets: To find the path to the data do the following (assuming everything installed properly) and replace ``__init__.py`` with ``data``

	``python -c "import dlatk; print(dlatk.__file__)"``

6. mysqlclient does not install in conda

	``conda install -c bioconda mysqlclient``