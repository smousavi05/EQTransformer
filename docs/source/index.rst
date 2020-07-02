.. figure:: figures/logo.png
    :width: 600px
    :align: left
    :alt: logo.png

.. EQTransformer documentation master file, created by
   sphinx-quickstart on Sat Jun 27 19:32:05 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EQTransformer's documentation!
=========================================

**EQTransformer** is an AI-based earthquake signal detector and phase (P&S) picker based on a deep neural network with an attention mechanism. It has a hierarchical architecture specifically designed for earthquake signals. **EQTransformer** has been trained on global seismic data and can perform detection and arrival time picking simultaneously. In addition to the prediction probabilities, it can also provides model uncertainties.   
 
The ``EQTransformer`` python 3 package includes modules for downloading continuous seismic data, preprocessing, performing earthquake signal detection, and phase (P & S) picking using pre-trained models, building and testing new models, and performing a simple phase association. 

The following is the main reference of **EQTransformer**:

* Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L.Y., Beroza, G.C., "Earthquake Transformer: An Attentive Deep-learning Model for Simultaneous Earthquake Detection and Phase Picking ". Nature Communications, (2020).


Github development page:
------------------------
https://github.com/smousavi05/EQTransformer


Contents
========
.. toctree::
    :numbered:
    :maxdepth: 2

    overview.rst
    installation.rst
    tutorial.rst
    copyrightlicense.rst
    developers.rst
    references.rst

Indices and tables
------------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
