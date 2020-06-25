.. EQTransformer documentation master file, created by
   sphinx-quickstart on Thu Apr 18 15:34:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: figures/
    :scale: 50 %
    :alt: Logo

Welcome to EQTransformer's Documentation!
==================================

EQTransformer is a `Tensorflow <https://www.tensorflow.org/>`_ based library for earthquake signal detection and phase picking using an attentive deep-learning model.
The ``EQTransformer`` python 3 package includes modules for downloading continuous seismic data, preprocessing, performing earthquake signal detection and phase (P & S) picking using pre-trained models, building and testing new models, and performing a simple phase association. 


.. figure:: figures/
    :align: center

Github development page:
------------------------

https://github.com/smousavi05/EQTransformer


Installation
------------
EQTransformer can be installed directly from pypi as:

.. code-block:: shell

    pip install EQTransformer



Citing EQTransformer
=============
To cite EQTransformer, use the following reference:

* 


Contents
========
.. toctree::
    :maxdepth: 1

    overview.rst
    examples/downloading_continuous_data
    examples/performing_detection&picking
    examples/visualizing
    examples/association
    examples/model_building
    documentation.rst
    copyrightlicense.rst
    developers.rst
    references.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

