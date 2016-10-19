.. _tut_dla:
==============
DLATK Tutorial
==============

In Differential Language Analysis (DLA) we correlate patterns in language with other characteristics such as gender, or voting results.  We may look at text broken down by user, county or individual message among other things.  You can see more about the conceptual aspect of DLA in this `Youtube Video <https://www.google.com/url?q=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DZdTeDED9h-w>`_ and in this journal paper, `Toward Personality Insights from Language Exploration in Social Media <http://wwbp.org/papers/sam2013-dla.pdf>`_.

In this tutorial we will walk you through the process of running DLA using the dlatk.py interface tool. Before running DLA here are some questions to ask yourself:

* What text am I using?
	* twitter, facebook, blogs
* What relationships am I looking at?
	* LIWC category prevalence VS voting habits
	* Usage of the word “I” VS narcissism score on personality quiz
	* What words correlate with extraversion score on a personality quiz
* How will I group my data?
	* by message, by user, by country, by hour

Answers for this tutorial:

* We will be using text from blogs (a subset of data from `this corpus <http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm>`_).
* We will look at age, gender and personality and how they correlate with LIWC and Facebook topics.
* We will group the data at the user level. 

**Difference between mysql and dlatk**: You will be using our infrastructure code (which is primarily written in Python) and mysql. If you are new to using a terminal this can be confusing. Step 0 shows you how to install the infrastructure code. You will start every command with **./dlatk.py**. As the name suggests, this is an interface to a much larger set of code called Feature Worker. The *.py* tells you that this is a Python file. On the other hand, **mysql** is a database management system. You access the mysql command line interface by typing **mysql** in your terminal. Mysql commands only work in the mysql command line interface, just as fwInterface only works in the terminal. Anytime you are asked to run a mysql command you must first type *mysql*. To exit you simply type *exit*. You should also note that all mysql commands end with a semicolon. When running this tutorial it might be helpful to have two terminal windows open, one for mysql and another for dlatk. 

STEP 0 - Prepare
================

The text we will use is the mysql database **dla_tutorial** and in the table **msgs**.  We will copy it to a new table using the following **mysql** commands:	

**Replace xxx with your initials so that your message table name is unique and your results don’t get messed up with anyone else’s results!!** 

.. code-block:: mysql

 USE dla_tutorial;
 CREATE TABLE msgs_xxx LIKE msgs; 
 INSERT INTO msgs_xxx SELECT * FROM msgs;

The relationships we will look at are ngram usage versus age and gender.  We will also look at Facebook topics versus age and gender and personality scores.  This outcome data is stored in the table **dla_tutorial.blog_outcomes** in the columns **age** and  **gender**.  

We will group our data by user. You can see both the message table and the outcome table have a column called **user_id**. 

STEP 1 - Feature Extraction
===========================

Generate n-gram Features
------------------------
This step generates a quantitative summary of a body of text.  It basically does word/n-gram counts on a group by group basis.  It also normalizes by group, so at the end of this you can answer questions like “what proportion of words used by USER 234459 were the word ‘the’”.

.. code-block:: bash

	##EXPECTED OUTPUT TABLES 
	#feat$1gram$msgs_xxx$user_id$16to16
	#feat$1gram$msgs_xxx$user_id$16to16$0_001
	./dlatk.py -d dla_tutorial -t msgs_xxx -c user_id --add_ngrams -n 1 --feat_occ_filter --set_p_occ 0.001 --group_freq_thresh 500

Brief descriptions of the flags:

* :doc:`../fwinterface/fwflag_d`: the database we are using
* :doc:`../fwinterface/fwflag_t`: the table inside the database where our text lives
* :doc:`../fwinterface/fwflag_c`: the table column we will be grouping the text by
* :doc:`../fwinterface/fwflag_add_ngrams`: the flag which starts the ngram extraction process
* :doc:`../fwinterface/fwflag_n`: the value or values for *n* in ngrams
* :doc:`../fwinterface/fwflag_feat_occ_filter`: This tells us to ignore features which occur in a small percentage of groups
* :doc:`../fwinterface/fwflag_set_p_occ`: The percentage value for the feature occurrence filter 
* :doc:`../fwinterface/fwflag_group_freq_thresh`: Ignore groups which do not contain a certain number of words
 
.. code-block:: bash

	##OTHER COMMAND OPTIONS
	./dlatk.py -d <database> -t <message_table> -c <group_data_column> --add_ngrams -n 1 2 3 --combine_feat_tables 1to3gram
	
	##FOLLOWED BY
	./dlatk.py -d <database> -t <message_table> -c <group_data_column> -f <feature table> --feat_occ_filter --set_p_occ <pocc> --group_freq_thresh <gft>

To view the columns in your feature table use the following **mysql** command:

.. code-block:: mysql

	describe feat$1gram$msgs_xxx$user_id$16to16;

This will give you the following output

.. code-block:: mysql

	+------------+---------------------+------+-----+---------+----------------+
	| Field      | Type                | Null | Key | Default | Extra          |
	+------------+---------------------+------+-----+---------+----------------+
	| id         | bigint(16) unsigned | NO   | PRI | NULL    | auto_increment |
	| group_id   | varchar(45)         | YES  | MUL | NULL    |                |
	| feat       | varchar(28)         | YES  | MUL | NULL    |                |
	| value      | int(11)             | YES  |     | NULL    |                |
	| group_norm | double              | YES  |     | NULL    |                |
	+------------+---------------------+------+-----+---------+----------------+

Summary of the columns:

* **id**: numeric value of a sql table row
* **group_id**: user ids from your message table
* **feat**: the 1grams
* **value**: the number of times the 1gram occurred within the group
* **group_norm**: the value divided by the total number of features for this group

To view the features tables use the following command in **mysql**. This will show every column value in the first 10 rows.

.. code-block:: mysql

	mysql> select * from dla_tutorial.feat$1gram$msgs_xxx$user_id$16to16 limit 10;
	+----+----------------------------------+-----------+-------+----------------------+
	| id | group_id                         | feat      | value | group_norm           |
	+----+----------------------------------+-----------+-------+----------------------+
	|  1 | 003ae43fae340174a67ffbcf19da1549 | neighbors |     1 | 0.000260010400416017 |
	|  2 | 003ae43fae340174a67ffbcf19da1549 | all       |    15 |  0.00390015600624025 |
	|  3 | 003ae43fae340174a67ffbcf19da1549 | jason     |     1 | 0.000260010400416017 |
	|  4 | 003ae43fae340174a67ffbcf19da1549 | <newline> |     5 |  0.00130005200208008 |
	|  5 | 003ae43fae340174a67ffbcf19da1549 | caused    |     1 | 0.000260010400416017 |
	|  6 | 003ae43fae340174a67ffbcf19da1549 | beware    |     1 | 0.000260010400416017 |
	|  7 | 003ae43fae340174a67ffbcf19da1549 | bull      |     1 | 0.000260010400416017 |
	|  8 | 003ae43fae340174a67ffbcf19da1549 | focus     |     1 | 0.000260010400416017 |
	|  9 | 003ae43fae340174a67ffbcf19da1549 | yellow    |     1 | 0.000260010400416017 |
	| 10 | 003ae43fae340174a67ffbcf19da1549 | four      |     3 |  0.00078003120124805 | 
	+----+----------------------------------+-----------+-------+----------------------+

You can also compare the sizes of the two tables to see the effect of --feat_occ_filter:

.. code-block:: mysql

	mysql> select count(distinct feat) from dla_tutorial.feat$1gram$msgs_xxx$user_id$16to16;
	+----------------------+
	| count(distinct feat) |
	+----------------------+
	|                65593 |
	+----------------------+

	mysql> select count(distinct feat) from dla_tutorial.feat$1gram$msgs_xxx$user_id$16to16$0_1;
	+----------------------+
	| count(distinct feat) |
	+----------------------+
	|                 1872 |
	+----------------------+

What would you expect the count to be if you had used a set_p_occ value of 0.01? 

Given the definition of group norm above, what would you expect to get if you summed all of the group norms for a single group? Verify your answer with the following **mysql** command:

.. code-block:: mysql

	select group_id, sum(group_norm) from dla_tutorial.feat$1gram$msgs_xxx$user_id$16to16 group by group_id limit 10;

Generate Lexicon (topic) Features
---------------------------------
This step **uses the 1gram feature table** that was used in step 1a in addition to some topic definitions.  It calculates a value that characterizes how strongly each topic was present in the text of a given group.  Sometimes this is as simple as aggregating counts.  Sometimes there is a weighting factor involved.  LIWC2007 and many other topic tables exists in the permaLexicon database schema. `Go here <http://www.liwc.net/>`_ for more information on LIWC (Linguistic Inquiry and Word Count). First, lets look at the LIWC2007 lex table:

.. code-block:: mysql

	mysql> select * from permaLexicon.LIWC2007 limit 10;
	+----+--------+----------+--------+
	| id | term   | category | weight |
	+----+--------+----------+--------+
	|  1 | y'all  | PPRON    |      1 |
	|  2 | ive    | PPRON    |      1 |
	|  3 | weve   | PPRON    |      1 |
	|  4 | she'll | PPRON    |      1 |
	|  5 | you'd  | PPRON    |      1 |
	|  6 | thoust | PPRON    |      1 |
	|  7 | mine   | PPRON    |      1 |
	|  8 | his    | PPRON    |      1 |
	|  9 | shes   | PPRON    |      1 |
	| 10 | theyd  | PPRON    |      1 |
	+----+--------+----------+--------+

Every lex table will have the columns id, term, category and weight. Since LIWC is an unweighted lexica the weight column is set to 1.

.. code-block:: bash

	# EXPECTED OUTPUT TABLE
	# feat$cat_LIWC2007$msgs_xxx$user_id$16to16
	./dlatk.py -d dla_tutorial -t msgs_xxx -c user_id --add_lex_table -l LIWC2007

Or we could use a weighted, data driven lexicon like our 2000 Facebook topics. These topics were created from Facebook data using Latent Dirichlet allocation (LDA). `Go here <https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`_ for more info on LDA. Also see our :doc:`tut_lda`. The Facebook topic table in permaLexicon looks like

.. code-block:: mysql

	mysql> select * from permaLexicon.met_a30_2000_cp limit 10;
	+----+---------+----------+--------------------+
	| id | term    | category | weight             |
	+----+---------+----------+--------------------+
	|  1 | ce      | 344      |  0.000162284972412 |
	|  2 | concept | 344      |  0.000556947925369 |
	|  3 | cough   | 344      | 0.0000711541198235 |
	|  4 | bring   | 344      |   0.00570741964554 |
	|  5 | finest  | 344      |  0.000520020800832 |
	|  6 | human   | 344      | 0.0000762679547477 |
	|  7 | winds   | 344      |   0.00839234198794 |
	|  8 | faster  | 344      |  0.000218674830527 |
	|  9 | halfway | 344      |  0.000872790748418 |
	| 10 | blow    | 344      |    0.0120238095238 |
	+----+---------+----------+--------------------+

The main differences to notice are the category names and the weights. Since this lexica was produced using a data driven approach we make no attempt to label the categories (for example, in LIWC above we see the category 'personal pronouns'). Also, this lexica contains weight in the form of conditional probabilities. We now apply this to our message set:

.. code-block:: bash

	# EXPECTED OUTPUT TABLE
	# feat$cat_met_a30_2000_cp$msgs_xxx$user_id$16to16
	./dlatk.py -d dla_tutorial -t msgs_xxx -c user_id --add_lex_table -l met_a30_2000_cp --weighted_lexicon

Brief descriptions of the flags:

* :doc:`../fwinterface/fwflag_add_lex_table`: 
* :doc:`../fwinterface/fwflag_l`: 
* :doc:`../fwinterface/fwflag_weighted_lexicon`: 

Note -  for *LIWC2007* we are NOT using weights, but we are for *met_a30_2000_cp*.
Note - dlatk pieces together the expected name of the 1gram table using the information you give it in the -d, -t, and -c options 
Note - in the table name *met_a30_2000_cp*, met stands for messages english tokenizen, a30 stands for alpha = 30 (a tuning parameter in the LDA process) and 2000 means there are 2000 topics.

In general use the following syntax (*permaLexicon* is a database where all of our lexica are stored):

.. code-block:: bash

	## GENERAL SYNTAX FOR CREATING LEXICON FEATURE TABLES
	./dlatk.py -d <db> -t <msg_tbl> -c <grp_col> --add_lex_table -l <topic_tbl_from_permalexicon> [--weighted_lexicon]

Again, you can view the tables with the following **mysql** commands:

.. code-block:: mysql

	select * from dla_tutorial.feat$cat_met_a30_2000_cp_w$msgs_xxx$user_id$16to16 limit 10;
	select * from dla_tutorial.feat$cat_LIWC2007$msgs_xxx$user_id$16to16 limit 10;

What should the group norms sum to for a single group in the lexicon tables? Will this be the same as above? Why or why not?

.. code-block:: mysql

	select group_id, sum(group_norm) from dla_tutorial.feat$cat_met_a30_2000_cp_w$msgs_xxx$user_id$16to16 group by group_id limit 10;
	select group_id, sum(group_norm) from dla_tutorial.feat$cat_LIWC2007$msgs_xxx$user_id$16to16 group by group_id limit 10;

STEP 2 - Insights (DLA): Correlate features with outcomes
=========================================================

This step takes the quantified/summarized text and examines/uses relationships with information about the group.  One basic output is a correlation matrix in html format. You may need to download a program such as WinSCP to transfer the output files from our server to your computer in order to view the output.  

.. code-block:: bash

	./dlatk.py -d dla_tutorial -t msgs_xxx -c user_id \ 
	-f 'feat$cat_LIWC2007$msgs_xxx$user_id$16to16' \ 
	 --outcome_table blog_outcomes \ 
	 --group_freq_thresh 500 \ 
	 --outcomes age gender \ 
	 --output_name xxx_output --rmatrix --sort --csv

Brief descriptions of the flags:

* :doc:`../fwinterface/fwflag_outcome_table`: 
* :doc:`../fwinterface/fwflag_outcomes`: 
* :doc:`../fwinterface/fwflag_rmatrix`: 
* :doc:`../fwinterface/fwflag_sort`: 
* :doc:`../fwinterface/fwflag_csv`:

Output will be written to the file **xxx_output.csv** and **xxx_output.html**. The csv output should look like 

.. code-block:: bash

	feature,age,p,N,freq,gender,p,N,freq
	ACHIEV,0.10453337969466858,1.2486251420175023,499,24061,-0.1327959917320303,0.18924871053777773,499,24061
	ADVERBS,-0.097823107908957693,1.8490497097147072,499,77661,0.091427449910103736,2.6369379754861826,499,77661
	AFFECT,-0.060118741047985133,11.519149773307243,499,133155,0.094864627490032188,2.1840596807077146,499,133155

The HTML file should look like this when opened in a browser:
Attach:rmatrix_output.png

In this example, positive value for age correlates with older age, and negative correlates with younger. Similarly, a positive value for gender indicates correlation with female, and a negative value correlates with male. 
Or using the Facebook topics and creating topic tag clouds:

.. code-block:: bash

	./dlatk.py -d dla_tutorial -t msgs_xxx -c user_id \ 
	-f 'feat$cat_met_a30_2000_cp_w$msgs_xxx$user_id$16to16' \ 
	 --outcome_table blog_outcomes  --group_freq_thresh 500 \ 
	 --outcomes age gender --output_name xxx_output \ 
	 --topic_tagcloud --make_topic_wordcloud --topic_lexicon met_a30_2000_freq_t50ll \ 
	--tagcloud_colorscheme bluered

Brief descriptions of the flags:

* :doc:`../fwinterface/fwflag_topic_tagcloud`: 
* :doc:`../fwinterface/fwflag_make_topic_wordcloud`: 
* :doc:`../fwinterface/fwflag_topic_lexicon`: 
* :doc:`../fwinterface/fwflag_tagcloud_colorscheme`: 

The following line will be printed to the screen:

.. code-block:: bash

	Yielding norms with zeros (500 groups * 2000 feats).

This tells us that we have 500 users (since our -c field is user_id) each with 2000 features. The 2000 features comes from the fact that we are working with 2000 Facebook topics.  Looking in MySQL we see that we have 500 users total in our dataset:

.. code-block:: mysql

	mysql> select count(distinct user_id) from msgs_xxx;
	+-------------------------+
	| count(distinct user_id) |
	+-------------------------+
	|                     500 |
	+-------------------------+

This means that every user in our dataset passes the group frequency threshold, i.e., each user has at least 500 words. If we were to set the group frequency threshold to 5000 we would see:

.. code-block:: bash

	Yielding norms with zeros (125 groups * 2000 feats).

Output will be written to the file **xxx_output_topic_tagcloud.txt**. The topic tagcloud output will be in a directory called *xxx_output_topic_tagcloud_wordclouds*

|| border=1
||! Topics most correlated with outcome !||
||! Outcome ||! Positive Correlation ||! Negative Correlation ||
|| Gender ||  Attach:gender_pos.png || Attach:gender_neg.png ||
|| Age    ||  Attach:age_pos.png ||  Attach:age_neg.png ||

Here is the general syntax for some other commands:

.. code-block:: bash

	####MAKE WORDCLOUDS
	./dlatk.py -d <db> -t <msg_tbl> -c <grp_col> -f <feat_tbl>  \ 
	 --outcome_table <table_with_group_info>  \ 
	 --outcomes <list of outcomes separated by spaces>  \ 
	 --output_name <desired_output_name> --tagcloud --make_wordclouds 

.. code-block:: bash

	####MAKE TOPIC WORDCLOUDS 
	./dlatk.py -d <db> -t <msg_tbl> -c <grp_col> -f <feat_tbl>  \ 
	 --outcome_table <table_with_group_info>  \ 
	 --outcomes <list of outcomes separated by spaces>  \ 
	 --output_name <desired_output_name> --topic_tagcloud --make_topic_wordcloud 
	 --topic_lexicon <lex_table>


Continuing on...
================
More information about dlatk's interface can be found in the following places: 

* :doc:`dlatkinterface_ordered`
* Next tutorial: :doc:`tut_pred`