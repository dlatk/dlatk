.. _fwflag_n_components:
==============
--n_components
==============
Switch
======

--n_components

Description
===========

Specifies the number of clusters when using --fit_reducer. For PCA one can also specify None or mle. For NMF and SPARSEPCA one can also specify None

Argument and Default Value
==========================

When n_components is not present, the default number of clusters is 24.

Details
=======

This switch changes the following parameter in each of the models (specified by :doc:`fwflag_model`): NMF: n_components
PCA: n_components
SPARSEPCA: n_components
LDA: nb_topics
KMEANS: n_clusters
SPECTRAL: n_clusters
GMM: n_components
This switch is ignored and no default value is set:
DBSCAN

Other Switches
==============

Note: There are other required switches for :doc:`fwflag_fit_reducer` 
Required Switches:
:doc:`fwflag_fit_reducer` :doc:`fwflag_model` nmf, pca, sparsepca, lda, kmeans, dbscan, spectral or gmm

Example Commands
================
.. code:doc:`fwflag_block`:: python


 # General syntax
 ./fwInterface.py :doc:`fwflag_d` <DATABASE> :doc:`fwflag_t` <TABLE> :doc:`fwflag_c` <> :doc:`fwflag_f` <FEATURE_TABLE> :doc:`fwflag_fit_reducer` :doc:`fwflag_model` <MODEL_NAME> :doc:`fwflag_n_components` N

 # Example command
 ./fwInterface.py :doc:`fwflag_d` primals :doc:`fwflag_t` primals_new :doc:`fwflag_c` dp_id :doc:`fwflag_f` 'feat$1to3gram$primals_new$dp_id$16to1$0_0001' :doc:`fwflag_fit_reducer` :doc:`fwflag_model` spectral :doc:`fwflag_n_components` 36
