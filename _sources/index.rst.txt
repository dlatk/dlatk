.. DLATK documentation master file, created by
   sphinx-quickstart on Wed Sep  7 15:59:11 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Differential Language Analysis ToolKit
--------------------------------------

DLATK is an end to end human text analysis package, specifically suited for social media and social scientific applications. It is written in Python 3 and developed by the World Well-Being Project at the University of Pennsylvania and Stony Brook University. It contains:

* feature extraction
* part-of-speech tagging
* correlation
* prediction and classification
* mediation 
* dimensionality reduction and clustering
* wordcloud visualization

DLATK can utilize:

- `HuggingFace <https://huggingface.co/>`_ for transformer language models
- `Mallet <http://mallet.cs.umass.edu/>`_ for creating LDA topics
- `Stanford Parser <http://nlp.stanford.edu/software/lex-parser.shtml>`_
- `CMU's TweetNLP <http://www.cs.cmu.edu/~ark/TweetNLP/>`_ 
- `pandas <http://pandas.pydata.org/>`_ dataframe output


Getting Started
---------------

.. toctree::
   :maxdepth: 1

   install
   Github Repo <https://github.com/dlatk/dlatk/>
   Getting started in Colab <https://colab.research.google.com/drive/10WMCmnKzwywZR7s2et5xx9CcoWBNmhLY?usp=sharing>
   tutorials
   datasets
   dlatkinterface_ordered
   papers

Citations
---------

If you use DLATK in your work please cite the following `paper <https://wwbp.org/papers/DLATK_Differential_Language_Analysis_ToolKit.pdf>`_:

.. code-block:: bash

      @InProceedings{DLATKemnlp2017,
        author =  "Schwartz, H. Andrew
            and Giorgi, Salvatore
            and Sap, Maarten
            and Crutchley, Patrick
            and Eichstaedt, Johannes
            and Ungar, Lyle",
        title =   "DLATK: Differential Language Analysis ToolKit",
        booktitle =  "Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
        year =    "2017",
        publisher =  "Association for Computational Linguistics",
        pages =   "55--60",
        location =   "Copenhagen, Denmark",
        url =  "http://aclweb.org/anthology/D17-2010"
      }


More Information
----------------

* `DLATK GitHub page <http://www.github.com/dlatk/dlatk>`_
* `DLATK at DockerHub <https://hub.docker.com/r/dlatk/dlatk/>`_
* `World Well-Being Project <http://www.wwbp.org>`_
* `Human Language Analysis Beings (HLAB) at Stony Brook <http://hlab.cs.stonybrook.edu/>`_
* `Computational Psychology & Well-Being Lab (CPWB) at Stanford University <https://cpwb.stanford.edu/>`_
* :doc:`modules`
* :doc:`changelog`

DLATK is licensed under a `GNU General Public License v3 (GPLv3) <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.

