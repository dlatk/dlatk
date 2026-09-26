*****************
Packaged Datasets
*****************

All datasets are available on our `github <http://www.github.com/dlatk/dlatk>`_ page, the `World Well-Being Project <http://www.wwbp.org>`_ site and via the pip install.

Note: some lexica and datasets are distributed on more restrictive licenses than DLATK. Please review each before use.


Language Data
=============

Blog Authorship Corpus
----------------------
A subset of blog posts from `this <http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm>`_ dataset collected by `J. Schler, M. Koppel, S. Argamon and J. Pennebaker <http://u.cs.biu.ac.il/~schlerj/schler_springsymp06.pdf>`_. This subset contains all posts from a random set of 1000 users. Shared with permission from Moshe Koppel.

* `[.zip] <http://wwbp.org/downloads/public_data/blogCorpus.zip>`_ 
* MySQL: dla_tutorial.msgs, dla_tutorial.blog_outcomes

Lexica
======

Age and Gender Lexica 
---------------------
Our data-driven age and gender lexica were generated from about 97,000 Facebook, Blogger and Twitter users. 

* `[.zip] <http://wwbp.org/downloads/public_data/emnlp2014_ageGenderLexica.zip>`_ 
* MySQL: permaLexicon.dd_emnlp14_ageGender
* `Link to publication <http://wwbp.org/publications.html#p3>`_

PERMA Lexicon
-------------
Our lexicon to predict well-being as measured through PERMA scales. 

* `[.zip] <http://wwbp.org/downloads/public_data/permaV3_dd.zip>`_ 
* MySQL: permaLexicon.dd_permaV3
* `Link to publication <http://wwbp.org/publications.html#p76>`_
* `[Usage license] <http://wwbp.org/downloads/public_data/ddpermav3_license.txt>`_

Spanish PERMA Lexicon
---------------------
Our lexicon to measure PERMA in Spanish, derived from Spanish tweets annotated with PERMA. 

* `[.zip] <http://wwbp.org/downloads/public_data/dd_sperma_v1.zip>`_
* MySQL: permaLexicon.dd_sperma_v2
* `Link to publication <http://wwbp.org/publications.html#p76>`_

Other Lexica
------------
Prospection Lexicon: Temporal Orientation: 

* `[.csv] <http://wwbp.org/downloads/public_data/temporalOrientationLexicon.csv>`_
* MySQL: permaLexicon.dd_PaPreFut
* `Link to publication <http://wwbp.org/publications.html#p76>`_

Affect and Intensity Lexicon: 

* `[.csv] <http://wwbp.org/downloads/public_data/ddIntAff.csv>`_
* MySQL: permaLexicon.dd_intAff
* `Link to publication <http://wwbp.org/publications.html#p76>`_


LDA Topics
==========

2000 Facebook Topics
--------------------

* Top 20 words per topic: `[.csv] <http://wwbp.org/downloads/public_data/2000topics.top20freqs.keys.csv>`_ `[Excel file] <http://wwbp.org/downloads/public_data/2000topics.top20freqs.keys.xls>`_
* MySQL: permaLexicon.met_a30_2000_cp and permaLexicon.met_a30_2000_freq_t50ll
* All words: `[.csv] <http://wwbp.org/downloads/public_data/wwbpFBtopics_freq.csv>`_
* Conditional probabilities `[.csv] <http://wwbp.org/downloads/public_data/wwbpFBtopics_condProb.csv>`_ (sparse matrix format)
* `Link to publication <http://wwbp.org/publications.html#p7>`_
