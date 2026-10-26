.. _tut_interactions:
=====================
DLA with Interactions
=====================

Running DLA with an interaction term
------------------------------------

* :doc:`../fwinterface/interaction`
* :doc:`../fwinterface/output_interaction_terms`

.. code-block:: bash

   dlatkInterface.py -d dla_tutorial -t msgs -c user_id \ 
   -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' \ 
   --outcome_table blog_outcomes  --group_freq_thresh 500 \ 
   --outcomes age --interaction gender \
   --csv --output_name interaction_output --output_interaction_terms  \ 
   --correlate 

The above command will produce a csv with three columns corresponding to the three variables in the regression model:

* **age**: the language variable
* **gender with age**: the interaction term alone
* **group_norm * gender from age**: the product term

For every feature we calculate a beta, p, N and a confidence interval for each of the three variables above: 

.. code-block:: bash

   feature,age,p,N,CI_l,CI_u,freq,gender with age,p,N,CI_l,CI_u,freq,group_norm * gender from age,p,N,CI_l,CI_u,freq
   0,0.0344849879812,0.375110412634,978,-0.0282628861297,0.0969621565127,57657,0.0204222202935,0.632378566764,978,-0.0423187330217,0.0830027366666,57657,0.017539415798,0.854171578743,978,-0.0451970546256,0.0801380815627,57657
   1,0.016872174598,0.683227405819,978,-0.0458631098209,0.0794748937429,184425,0.0190901461024,0.632378566764,978,-0.0436488652219,0.0816791773924,184425,0.0339621286241,0.718894793283,978,-0.0287859326858,0.0964435792616,184425
   10,-0.031671145902,0.420633811537,978,-0.0941709591925,0.0310773305901,49964,0.0193195458569,0.632378566764,978,-0.0434198157553,0.0819071265087,49964,-0.0109437945432,0.916875173924,978,-0.0735800930425,0.0517785038655,49964
   100,-0.0296604288467,0.576185788081,978,-0.0921758215433,0.0330878687188,51687,0.022243809343,0.632378566764,978,-0.0404994395195,0.0848123243954,51687,0.0625745873057,0.571561056997,978,-0.000112631240734,0.124771933759,51687

From this we can determine which language features have a significant interaction term by looking at the **group_norm * gender from age** columns. 

DDLA with an interactions
-------------------------

Sometimes called "post-hoc probing of interactions", here we build off of the above and include two DLA steps. The following command will:

#. Run DLA with the addition of an interaction term.
#. Run DLA, whitelisting on those features which have a significant interaction term from Step 1, only on users where gender = 1
#. Repeat Step 2 on users where gender = 0

We use the additional flag 

* :doc:`../fwinterface/interaction_ddla`

and assume a categorical variable encoded as 0 or 1. 

.. code-block:: bash

   dlatkInterface.py -d dla_tutorial -t msgs -c user_id \ 
   -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' \ 
   --outcome_table blog_outcomes  --group_freq_thresh 500 \ 
   --outcomes age --interaction_ddla gender \
   --csv --output_name interaction_output --output_interaction_terms  \ 
   --topic_tagcloud --make_topic_wordcloud --topic_lexicon met_a30_2000_freq_t50ll --tagcloud_colorscheme bluered \
   --correlate 

Four sets of wordclouds will be produced: 

* **INTER[age]**: the language variable
* **INTER[gender with age]**: the interaction term alone
* **[age]_1**: the language variable subsetted to users with gender = 1
* **[gender]_0**: the language variable subsetted to users with gender = 0

Using two continuous variables
------------------------------

The above commands assumed that one of the variables of interest (gender) was a categorical variable. We can also run over two continuous variables with some extra MySQL work. Here are the general steps given a continous interaction variable *foo*:

#. Standardize *foo*: subtract the mean and divide by the standard deviation
#. Create a new column in your MySQL outcome table *foo_binary* 
#. Set *foo_binary*  = 1 where the standardized *foo* >= 1
#. Set *foo_binary*  = 0 where the standardized *foo* <= -1
#. Use both :doc:`../fwinterface/interaction_ddla` and :doc:`../fwinterface/interaction` to distinguish the two as follows: 

.. code-block:: bash

   dlatkInterface.py -d dla_tutorial -t msgs -c user_id ... \ 
   --outcomes age --interaction foo --interaction_ddla foo_binary \
   ... 

The above command is similar to the "DDLA with an interactions" command except we now use :doc:`../fwinterface/interaction` for the continuous variable in Step 1 and :doc:`../fwinterface/interaction_ddla` for the categorical variable in Steps 2 and 3. 
