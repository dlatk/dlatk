.. _tut_adv_fe:
.. raw:: html

   <div class="AutoTOCdiv">

+-------------------------------------+
| **Contents** [`hide`_]              |
+-------------------------------------+
| `Switch`_                           |
| `Description`_                      |
|     `Argument and Default Value`_   |
|     `Details`_                      |
|     `Other Switches`_               |
|     `Example Commands`_             |
|         `Author`_                   |
|         `References`_               |
+-------------------------------------+

.. raw:: html

   </div>

Switch
------

--------------

–p\_correction

.. raw:: html

   <div class="vspace">

.. raw:: html

   </div>

Description
-----------

--------------

Specifies a p-value correction method (for multiple comparisons) in
`correlation`_ other than Bonferroni (which is turned off with
–`no\_bonferroni`_)

.. raw:: html

   <div class="vspace">

.. raw:: html

   </div>

Argument and Default Value
~~~~~~~~~~~~~~~~~~~~~~~~~~

--------------

Argument: method to use

.. raw:: html

   <div class="vspace">

.. raw:: html

   </div>

Details
~~~~~~~

--------------

Possible values include:
``simes, holm, hochberg, hommel, bonferroni, BH, BY, fdr, none``

``simes`` is built into featureWorker; anything else calls R’s stats
module, specifically the ``p_adjust`` command

.. raw:: html

   <div class="vspace">

.. raw:: html

   </div>

Other Switches
~~~~~~~~~~~~~~

--------------

Required Switches:

-  –`no\_bonferroni`_
-  –`correlate`_

Optional Switches:

-  None

.. raw:: html

   <div class="vspace">

.. raw:: html

   </div>

Example Commands
~~~~~~~~~~~~~~~~

--------------

.. raw:: html

   <div class="vspace">

.. raw:: html

   </div>

Author
^^^^^^

Patrick

.. raw:: html

   <div class="vspace">

.. raw:: html

   </div>

References
^^^^^^^^^^

-  https://en.wikipedia.org/wiki/Multiple_comparisons_problem
-  https