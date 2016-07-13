FEATURE WORKER v0.6.1
======================

This package offers end to end text analysis: feature extraction, correlation, 
mediation and prediction / classification. Please visit for more 
information

  http://wwbp.org
  http://wiki.wwbp.org

SETUP (Linux)
=====
 Feature Worker has been tested on Ubuntu 14.04.

 1. Install the required Ubuntu libraries. The package python-rpy2 can also be omitted if 	
 	you experience installation issues, thought his will limit some of the advanced methods in Feature Worker.
 	WARNING: This will install MySQL on your computer.  

 		xargs apt-get install < install/requirements.system

 2. Install python dependencies.

    	pip install -r install/requirements.txt

 3. Load NLTK corpus

    	python -c "import nltk; nltk.download('wordnet')"

 4. Install Stanford parser

 	Download the zip file from http://nlp.stanford.edu/software/lex-parser.shtml. 
 	Extract into ../FeatureWorker/Tools/StanfordParser/. Move 
 	../FeatureWorker/Tools/StanfordParser/oneline.sh into the folder you extracted:
 	../FeatureWorker/Tools/StanfordParser/stanford-parser-full*/.
    
 5. Install Tweet NLP v0.3 (ark-tweet-nlp-0.3)

 	Download the tgz file (for version 0.3) from http://www.cs.cmu.edu/~ark/TweetNLP/.
 	Extract this file into ../FeatureWorker/Tools/TwitterTagger/.

 6. (Optional) Install the IBM Wordcloud jar file. 

 	The default wordcloud module is installed in Step 2 via pip. This can be changed 
 	to the IBM wordcloud module which produces nicer wordclouds. To do this:
 	
 	 a.	You must sign up for a IBM DeveloperWorks account and download
 		ibm-word-cloud.jar. Place this file into ../FeatureWorker/lib/. 

 	 b.	Change line number 108 in ../FeatureWorker/lib/wordcloud.py from
 			wordcloud_algorithm='amueller'
 		to
 		    wordcloud_algorithm='ibm'

SETUP (OSX)
=====
 Feature Worker has been tested on OSX 10.11.

 1. Install pip.

 		sudo easy_install pip

 2. Install brew. See www.brew.sh for information.


 3. Install dependencies with brew. WARNING: This will install MySQL on your computer.

    	brew install $(<install/requirementsOSX.system)

 4. Install python dependencies. Note that there is an issue when running this 
 	on OSX El Capitan which can be fixed by adding '--ignore-installed six' to the
 	end of the following command. This issue does not seem to be an issue if 
 	anaconda is installed.

    	pip install -r install/requirementsOSX.txt

 5. Load NLTK corpus

    	python -c "import nltk; nltk.download('wordnet')"

 5. Install Stanford parser

 	Download the zip file from http://nlp.stanford.edu/software/lex-parser.shtml. 
 	Extract into ../FeatureWorker/Tools/StanfordParser/. Move 
 	../FeatureWorker/Tools/StanfordParser/oneline.sh into the folder you extracted:
 	../FeatureWorker/Tools/StanfordParser/stanford-parser-full*/.
    
 6. Install Tweet NLP v0.3 (ark-tweet-nlp-0.3)

 	Download the tgz file (for version 0.3) from http://www.cs.cmu.edu/~ark/TweetNLP/.
 	Extract this file into ../FeatureWorker/Tools/TwitterTagger/.

 7. (Optional) Install the IBM Wordcloud jar file. 

 	The default wordcloud module is installed in Step 2 via pip. This can be changed 
 	to the IBM wordcloud module which produces nicer wordclouds. To do this:
 	
 	 a.	You must sign up for a IBM DeveloperWorks account and download
 		ibm-word-cloud.jar. Place this file into ../FeatureWorker/lib/. 

 	 b.	Change line number 108 in ../FeatureWorker/lib/wordcloud.py from
 			wordcloud_algorithm='amueller'
 		to
 		    wordcloud_algorithm='ibm'

MYSQL CONFIGURATION
==============

 1. Feature Worker is highly dependent on MySQL. You must have this installed (see Step
	1 in SETUP (Linux) or Step 3 in SETUP (OSX)). 

 2. Any calls to fwInterface.py will open MySQL. With your database any text data 
 	must have two columns:
 		message: text data
 		message_id: unique numeric identifier for each message

 3. All lexicon tables are assumed to be in a database called permaLexicon. To change this
	you must edit line 90 in fwInterface.py:
		DEF_LEXICON_DB = 'permaLexicon'

EXAMPLE COMMANDS
==============

 1. Feature Extraction

 	Given a message set in the table tweetcollectiondb.messagestable, this extracts 
 	1grams, 2grams and 3grams and places them into a single table

		./fwInterface.py -d tweetcollectiondb -t messagestable -c user_id 
		--add_ngrams -n 1 2 3 --feat_occ_filter --set_p_occ 0.05 --combine_feat_tables 1to3gram

	Given a 1gram feature table feat$1gram$messages_r5k$user_id$16to16, this removes
	all features used in less than 5% of users.

		./fwInterface.py -d tweetcollectiondb -t messagestable -f 'feat$1gram$messages_r5k$user_id$16to16' 
		--feat_occ_filter --set_p_occ 0.05

 2. Correlation

 	Correlate 1grams with age while controlling for gender.

		./fwInterface.py -d tweetcollectiondb -t messagestable -c user_id -f 
		'feat$1gram$messages_r5k$user_id$16to16$0_05' --group_freq_thresh 500 --outcome_table user_data 
		--outcomes age --outcome_controls gender --rmatrix --csv --output_name TEST

CONTACT
=======

Please send bug reports, patches, and other feedback to

  Andy Schwartz (hansens@sas.upenn.edu) or Salvatore Giorgi (sgiorgi@sas.upenn.edu)
