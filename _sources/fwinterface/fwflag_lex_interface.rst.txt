.. _fwflag_lex_interface:
===============
--lex_interface
===============
Switch
======

--lex_interface

Description
===========

Override the argparser in dlatkInterface and send all arguments to lexInterface. lexInterface is often used to upload csv's to MySQL during the LDA process. See the :doc:`../tutorials/tut_lda` tutorial for more details. 

Details
=======

The full list of available flags in lexInterface:

.. code-block:: bash
	
	python lexInterface.py -h

	usage: lexInterface.py [-h] [-f FILENAME] [-g GFILE] [--sparsefile SPARSEFILE]
	                       [--weightedsparsefile WEIGHTEDSPARSEFILE]
	                       [--dicfile DICFILE] [--topicfile TOPICFILE]
	                       [--topic_csv] [--filter] [-n NAME] [-c CREATE] [-p]
	                       [--print_weighted] [--pprint] [-w WHERE] [-u UNION]
	                       [-i INTERSECT] [--super_topic SUPERTOPIC] [-r]
	                       [--depol] [--ungroup] [--compare COMPARE]
	                       [--annotate_senses SENSE_ANNOTATED_LEX]
	                       [--topic_threshold TOPICTHRESHOLD] [-a] [-l]
	                       [--corpus_examples] [--corpus_samples] [-e] [-d DB]
	                       [-t TABLE] [--lexicondb DB] [--corpus_term_field FIELD]
	                       [--corpus_message_field FIELD]
	                       [--corpus_messageid_field FIELD] [--min_word_freq NUM]
	                       [--lexicon_category CATEGORY] [--num_rand_messages NUM]

	On Features Class.

	optional arguments:
	  -h, --help            show this help message and exit

	:

	  -f FILENAME, --file FILENAME
	                        Lexicon Filename (default: None)
	  -g GFILE, --gfile GFILE   
	                        Lexicon Filename in google format (default: None)
	  --sparsefile SPARSEFILE   
	                        Lexicon Filename in sparse format (default: None)
	  --weightedsparsefile WEIGHTEDSPARSEFILE
	                        Lexicon Filename in weighted sparse format (default:
	                        None)
	  --dicfile DICFILE     Lexicon Filename in dic (LIWC) format (default: None)
	  --topicfile TOPICFILE
	                        Lexicon Filename in topic format (default: None)
	  --topic_csv, --weighted_file
	                        tells interface to use the topic csv format to make a
	                        weighted lexicon (default: False)
	  --filter              Allows lexicon filtering if True (default: False)
	  -n NAME, --name NAME  Existing Lexicon Table Name (will load) (default:
	                        None)
	  -c CREATE, --create CREATE
	                        Create a new lexicon table (must supply new lexicon
	                        name, and either -f, -g or -n) (default: None)
	  -p, --print           print lexicon to stdout (default csv format) (default:
	                        False)
	  --print_weighted      print lexicon to stdout (weighted csv format)
	                        (default: False)
	  --pprint              print lexicon to stdout as pprint output (default:
	                        False)
	  -w WHERE, --where WHERE   
	                        where phrase to add to sql query (default: None)
	  -u UNION, --union UNION   
	                        Unions two tables and uses the result as myLexicon
	                        (default: None)
	  -i INTERSECT, --intersect INTERSECT
	                        Intersects two tables and uses the result as myLexicon
	                        (default: None)
	  --super_topic SUPERTOPIC  
	                        Maps the current lexicon with a super topic mapping
	                        lexicon to make a super_topic (default: None)
	  -r, --randomize       Randomizes the categories of terms (default: False)
	  --depol               Depolarize the categories (removes +/-) (default:
	                        False)
	  --ungroup             places each word in its own category (default: False)
	  --compare COMPARE     Unions two tables and uses the result as myLexicon
	                        (default: None)
	  --annotate_senses SENSE_ANNOTATED_LEX
	                        Asks the user to annotate senses of words and creates
	                        a new lexicon with senses (new lexicon name is the
	                        parameter) (default: None)
	  --topic_threshold TOPICTHRESHOLD
	                        sets the threshold to use for a csv topicfile
	                        (default: None)
	  -a, --add_terms       Adds terms from the loaded lexicon to a given corpus
	                        (options below) (default: False)
	  -l, --corpus_lexicon  Load a lexicon based on finding words in a given
	                        corpus (BETA) (options below) (default: False)
	  --corpus_examples     Find example instances of words in the given corpus
	                        (using rlike; equal number for all words) (default:
	                        False)
	  --corpus_samples      Find sample of matches for lexicon. (default: False)
	  -e, --expand_lexicon  Expands the lexicon to more terms. (default: False)

	Terms OR Corpus Lexicon Options:

	  -d DB, --corpus_db DB
	                        Corpus database to use [default: dla_tutorial]
	  -t TABLE, --corpus_table TABLE
	                        Corpus table to use [default: msgs]
	  --lexicondb DB        The database which stores all lexicons. (default:
	                        dlatk_lexica)
	  --corpus_term_field FIELD 
	                        field of the corpus table that contains terms (lexicon
	                        table always uses 'term') [default: term]
	  --corpus_message_field FIELD
	                        field of the corpus table that contains the actual
	                        message [default: message]
	  --corpus_messageid_field FIELD
	                        field of the table that contains message ids (set to
	                        '' to not use group by [default: message_id]
	  --min_word_freq NUM   minimum number of instances to include in lexicon (-l
	                        option) [default: 1000]
	  --lexicon_category CATEGORY
	                        category in lexicon to get random samples from
	                        (default: None)
	  --num_rand_messages NUM   
	                        number of random messages to select when getting
	                        samples from lexicon category (default: 100)




Example Commands
================

Upload the topic given word probability distributions generated during LDA. This creates a table in `dlatk_lexica` called `msgs_lda_cp`.

.. code-block:: bash

	dlatkInterface.py --lex_interface --topic_csv  \ 
	--topicfile=/home/user/lda_tutorial/msgs_lda_tok_lda.lda_topics.topicGivenWord.csv  \ 
	-c msgs_lda_cp




