.. _tut_trns:
=================================
Transformers in DLATK
=================================

Transformers form the building blocks of many modern NLP models including ChatGPT and BARD. Modern language models are multiple layers of these transformers stacked on top of each other, and trained on huge corpora of texts.
These transformers-based language models produce numeric vectors representing words. The word vectors have found to be capturing strong semantic information and syntactic information. 

Huggingface (an open source research community) hosts these transformers based language models which can be accessed through the `transformers` library. The catalog of transformers models available through huggingface can be found (here)[https://huggingface.co/models] 

DLATK enables extraction of any huggingface transformers features from messages. These features are stored, as you might expect, in feature tables!

Prerequisites
-------------

You must have a ``msgs`` table in MySQL containing a ``message`` column.

You must also ensure that punkt is installed for NLTK. This can be accomplished with the following command:

.. code-block:: bash

	python -m nltk.downloader punkt

1. Adding Transformers features
-----------------------

For the purpose of this tutorial, we are going to extract BERT features. BERT is one of the first transformer-based language models which has been widely used for many tasks. Recommended reading about BERT:

* `Illustrated BERT <https://jalammar.github.io/illustrated-bert/>`_

The simplest way to add BERT embeddings, accepting all defaults, is with the :doc:`../fwinterface/fwflag_add_emb` flag:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id --add_emb_feat --emb_model bert-base-uncased

Note that the BERT features this adds will be aggregated at several levels:

* Layers
* Words
* Messages

BERT layers are aggregated to produce a single vector representation of a word. Words are aggregated to produce message-level vector representations. Messages are aggregated to produce vector representations at the level of the grouping factor (:doc:`../fwinterface/fwflag_c`). In the command above, this means messages are aggregated to produce user-level vector representations.

This command uses all BERT defaults. However, it is possible to customize BERT features in a number of ways:

* :doc:`../fwinterface/fwflag_emb_model`
* :doc:`../fwinterface/fwflag_emb_layers`
* :doc:`../fwinterface/fwflag_emb_layer_aggregation`
* :doc:`../fwinterface/fwflag_emb_msg_aggregation`

In the following subsections, we discuss these flags in more detail.

--emb_model
^^^^^^^^^^^^

The most important option for Transformers is the choice of model using the :doc:`../fwinterface/fwflag_emb_model` flag. Any `HuggingFace pretrained models <https://huggingface.co/transformers/models>`_ may be used here. By default, BERT features are extracted using the ``bert-base-uncased`` model; you can specify other models like so:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id --add_emb_feat --emb_model bert-large-uncased


--emb_layers
^^^^^^^^^^^^^

Transformers-based language models produces multiple layers of embeddings (because it is a *deep* network, so each network layer produces a layer of embeddings!). Roughly speaking, later layers embed more abstract representations of features, while earlier layers represent more concrete ones. It is typical to combine some or all of these layers in order to capture this breadth of representation:

.. image:: ../_static/bert-layers.png

To specify which layers you want to aggregate over, use the :doc:`../fwinterface/fwflag_emb_layers` flag. This flag takes as arguments the indexes of each layer you want to keep. For example, we might run the following to keep the last two layers:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id --add_emb_feat --emb_model bert-base-uncased --emb_layers 11 12

There are 12 layers in the bert-base-uncased model. The 0th index corresponds to the static word embeddings, while 1st through 12th index correspond to the respective layers.

Aggregation
^^^^^^^^^^^

As discussed above, there are two levels of aggregation when adding transformers features: layer, and message. These can be specified with the following flags:

* :doc:`../fwinterface/fwflag_emb_layer_aggregation`
* :doc:`../fwinterface/fwflag_emb_msg_aggregation`

It is important when running these aggregations to remember that you're choosing a numpy method, and that it will be applied to the 0th axis (i.e., it will be applied across layers). Here's an example:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id --add_emb_feat --emb_model bert-large-uncased --emb_layer_aggregation mean --emb_msg_aggregation max

It is also possible to specify multiple aggregations. Aggregations will be applied in the order that you specify them.


3. Understanding Transformers Feature Table Names
-----------------------------------------

Transformers feature tables have names that might look confusing, but actually reveal all the details about how the features were computed. If you don't yet understand feature table naming conventions in DLATK, please read :doc:`tut_feat_tables` before continuing.

Let's say you have a BERT feature table called ``feat$bert_ba_un_meL10con$messages_en$user_id$16to16``. Here's how to interpret the segment ``bert_ba_un_meL10co``:


#. **bert**
#. **ba**\ se\_\ **un**\ cased
#. **me**\ an aggregated messages
#. **L**\ ayer 10
#. **con**\ catenated layers

Some of these may be repeated: each layer selected with :doc:`../fwinterface/fwflag_emb_layers`, for example, will appear in the name.

4. Using Transformers features
----------------------

Let's say you've generated default BERT features with the following command:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id --add_emb_feat --emb_model bert-base-uncased

This will create the table ``feat$bert_ba_un_meL10con$msgs_xxx$user_id`` in the ``dla_tutorial`` database. How do you make use of these features?

The answer is, essentially, like any other feature table in DLATK! (See :doc:`tut_pred` if you don't know how to use feature tables.) For example, let's say you want to predict age from the ``blog_outcomes`` table in the ``dla_tutorial`` database (just like in :doc:`tut_pred`). This would look like:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id -f 'feat$bert_ba_un_meL10con$msgs_xxx$user_id' --outcome_table blog_outcomes --group_freq_thresh 500 --outcomes age --output_name xxx_age_output --nfold_test_regression --model ridgecv --folds 10

This will run a ridge regression model through 10-fold cross validation, predicting age, and using BERT embeddings as features in the model. You should see lots of output, ending with something like this:

.. code-block:: bash

	[TEST COMPLETE]
	
	{'age': {(): {1: {'N': 978,
                  'R': 0.6618386965904822,
                  'R2': 0.43803046030458836,
                  'R2_folds': 0.4233643081827411,
                  'mae': 4.619034298134363,
                  'mae_folds': 4.616423448270654,
                  'mse': 38.854440157161314,
                  'mse_folds': 38.820150245521624,
                  'num_features': 768,
                  'r': 0.6621098985283438,
                  'r_folds': 0.6694097453791333,
                  'r_p': 2.0416764169778933e-124,
                  'r_p_folds': 1.783371417616072e-11,
                  'rho': 0.7107187086007063,
                  'rho_p': 2.9543434601896134e-151,
                  'se_R2_folds': 0.02206729102836306,
                  'se_mae_folds': 0.11535169076175808,
                  'se_mse_folds': 1.859884969097588,
                  'se_r_folds': 0.01581679229802067,
                  'se_r_p_folds': 9.484973024855944e-12,
                  'se_train_mean_mae_folds': 0.20067386714575627,
                  'test_size': 105,
                  'train_mean_mae': 4.335374453708489,
                  'train_mean_mae_folds': 6.464315148935017,
                  'train_size': 873,
                  '{modelFS_desc}': 'None',
                  '{model_desc}': 'RidgeCV(alphas=array([1.e+03, 1.e-01, '
                                  '1.e+00, 1.e+01, 1.e+02, 1.e+04, 1.e+05]),   '
                                  'cv=None, fit_intercept=True, gcv_mode=None, '
                                  'normalize=False,   scoring=None, '
                                  'store_cv_values=False)'}}}}


Comparing these results against those from :doc:`tut_pred`, we can see that BERT features get a Pearson r of 0.6621, outperforming LDA topics + unigrams, which get an r of 0.6496.

There's a natural question we've glossed over here: what exactly do the BERT features look like? We can check the contents of the feature table in MySQL:

.. code-block:: bash

	mysql> SELECT * FROM feat$bert_ba_un_meL10con$msgs_xxx$user_id LIMIT 10;
	+----+----------+------+-----------------------+-----------------------+
        | id | group_id | feat | value                 | group_norm            |
        +----+----------+------+-----------------------+-----------------------+
        |  1 |      666 | 0me  |  -0.20481809973716736 |  -0.20481809973716736 |
        |  2 |      666 | 1me  |  -0.48483654856681824 |  -0.48483654856681824 |
        |  3 |      666 | 2me  |    1.1650058031082153 |    1.1650058031082153 |
        |  4 |      666 | 3me  |   -0.5072966814041138 |   -0.5072966814041138 |
        |  5 |      666 | 4me  |   0.40456074476242065 |   0.40456074476242065 |
        |  6 |      666 | 5me  |   -0.6585525274276733 |   -0.6585525274276733 |
        |  7 |      666 | 6me  | -0.019926181063055992 | -0.019926181063055992 |
        |  8 |      666 | 7me  |    0.2585161030292511 |    0.2585161030292511 |
        |  9 |      666 | 8me  |   -0.2901904881000519 |   -0.2901904881000519 |
        | 10 |      666 | 9me  |   -0.2664993405342102 |   -0.2664993405342102 |
        +----+----------+------+-----------------------+-----------------------+

The names of the ``feat`` column may seem a bit opaque at first, but they are simple to interpret: the number indicates the index of the dimension in the BERT embedding vector, while the ``me`` indicates that the message embeddings were aggregated using the mean. If you have specified multiple message aggregations, these will appear as separate features. Since BERT produces vectors of length 768, this means each ``group_id`` will have ``768 * [number of message aggregations]`` features. Each dimension of the aggregated BERT embedding vector then serves as a distinct feature in the predictive model.
