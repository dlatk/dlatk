.. _tut_conda:
============================================
Getting Started with WWBPY Conda Environment
============================================

Add `export PATH="/data/anaconda2/bin:$PATH"` to your .bashrc or .zshrc

then you should be able to run `source activate wwbpy` (can also be added to .bashrc or .bash_profile or whatever you want)

quickstart: http://conda.pydata.org/docs/test-drive.html

`source deactivate` should get you back to current env python

The anaconda directory is only writable by me right now (I'll probably make it root eventually), but anyone should be able to clone a conda env in their home directory (but check that it doesn't get too big). See http://conda.pydata.org/docs/using/envs.html#clone-an-environment

You may need to add this to your ~/.my.cnf under [client]:

  socket=/var/run/mysqld/mysqld.sock

`Conda: Myths and Misconceptions <https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/>`_