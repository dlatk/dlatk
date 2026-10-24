.. _tut_lda:
====================
DLATK LDA Interface
====================

**Note: These instructions introduce the new streamlined interface for LDA topic estimation. To use the old manual interface, see** :doc:`tut_lda_full`.

For a conceptual overview of LDA, see `this intro <http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/>`_

Step 0: Setup
----------------------------

Mallet:

* This tutorial uses `Mallet <http://mallet.cs.umass.edu/>`_. 
* Install to your home directory using the following website: http://mallet.cs.umass.edu/download.php
* NOTE - if you plan to be running on large datasets (~15M FB messages or similar) you may have to adjust parameters in your mallet script file.  See more info in the "Run LDA with Mallet" step.

PyMallet:

* Depending on your DLATK installation, you may also need to install pymallet with the following command: ``pip install dlatk-pymallet``

Step 1: (If necessary) Create sample tweet table
------------------------------------------------
If necessary, create a message table to run LDA on:

.. code-block:: mysql

	use dla_tutorial; 
	create table msgs_lda like msgs;
	insert into msgs_lda select * from msgs where rand()<(2/6);


Step 2: Generate a feature table
-----------------------------------------------------------------------------------------------------
This is a standard unigram feature table generation command.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_lda -c message_id --add_ngrams -n 1

Step 3: Estimate LDA topics
---------------------------
A minimal command for estimating LDA topics is shown below:

.. code-block:: bash

    dlatkInterface.py -d dla_tutorial -t msgs_lda -c message_id \
        -f 'feat$1gram$msgs_lda$message_id$16to16' \
        --estimate_lda_topics \
        --lda_lexicon_name my_lda_lexicon

However, it is important to realize that the command above will estimate LDA topics using PyMallet, which is in general *much* slower than Mallet. To use Mallet for topic estimation, you can use the following command:

.. code-block:: bash

    dlatkInterface.py -d dla_tutorial -t msgs_lda -c message_id \
        -f 'feat$1gram$msgs_lda$message_id$16to16' \
        --estimate_lda_topics \
        --lda_lexicon_name my_lda_lexicon \
        --mallet_path /path/to/mallet/bin/mallet

Be sure to replace ``/path/to/mallet/bin/mallet`` with the correct path to which you installed Mallet in Step 0.

It is good practice to refrain from storing the topics as a lexicon until after you have reviewed them. While the interim LDA estimation files are typically stored in your ``/tmp`` directory, you can specify a different directory to allow you to more easily review the topics you have estimated. The following command will store these files in the ``lda_files`` directory and prevent creating a topic lexicon:

.. code-block:: bash

    dlatkInterface.py -d dla_tutorial -t msgs_lda -c message_id \
        -f 'feat$1gram$msgs_lda$message_id$16to16' \
        --estimate_lda_topics \
        --save_lda_files lda_files
        --no_lda_lexicon \
        --mallet_path /path/to/mallet/bin

You can now review the ``.keys`` file in the ``lda_files`` directory to view the estimated topics and decide whether you should change any parameters (e.g., :doc:`../fwinterface/fwflag_num_stopwords` or :doc:`../fwinterface/fwflag_lda_alpha`).

An important difference between this new interface and the old one is that stop words are no longer derived from a static Mallet stoplist. Instead, DLATK will determine the most common terms in your feature table and remove them (by default, it sets the top 50 most frequent terms as stop words, but this can be controlled with :doc:`../fwinterface/fwflag_num_stopwords`). To disable stopping entirely, use :doc:`../fwinterface/fwflag_no_lda_stopping`.

There are several options you may wish to use with :doc:`../fwinterface/fwflag_estimate_lda_topics`:

* :doc:`../fwinterface/fwflag_mallet_path`
* :doc:`../fwinterface/fwflag_save_lda_files`
* :doc:`../fwinterface/fwflag_lda_lexicon_name`
* :doc:`../fwinterface/fwflag_no_lda_lexicon`
* :doc:`../fwinterface/fwflag_num_topics`
* :doc:`../fwinterface/fwflag_num_stopwords`
* :doc:`../fwinterface/fwflag_no_lda_stopping`
* :doc:`../fwinterface/fwflag_lda_alpha`
* :doc:`../fwinterface/fwflag_lda_beta`
* :doc:`../fwinterface/fwflag_lda_iterations`

Step 4: Extract features from lexicon
--------------------------------------

You’re now ready to start using the topic distribution lexicon

.. code-block:: bash

    dlatkInterface.py -d DATABASE -t MESSAGE_TABLE -c GROUP_ID \
        --add_lex_table -l my_lda_lexicon_cp --weighted_lexicon

Always extract features using the ``_cp`` lexicon. The ``_freq_t50ll`` lexicon is only used when generating topic_tagclouds: :doc:`../fwinterface/fwflag_topic_tagcloud` :doc:`../fwinterface/fwflag_topic_lexicon`.
