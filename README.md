
# EQTransformer

![PyPI](https://img.shields.io/pypi/v/EQTransformer?style=plastic)
![Conda](https://img.shields.io/conda/v/smousavi05/eqtransformer?style=plastic)
![Read the Docs](https://img.shields.io/readthedocs/eqtransformer?style=plastic)
![PyPI - License](https://img.shields.io/pypi/l/EQTransformer?style=plastic)
![PyPI - Downloads](https://img.shields.io/pypi/dd/EQTransformer?style=plastic)
![PyPI - Downloads](https://img.shields.io/pypi/dw/EQTransformer?style=plastic)
![PyPI - Downloads](https://img.shields.io/pypi/dm/EQTransformer?style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/smousavi05/EQTransformer?style=plastic)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/smousavi05/EQTransformer?style=plastic)
![GitHub stars](https://img.shields.io/github/stars/smousavi05/EQTransformer?style=social)
![GitHub followers](https://img.shields.io/github/followers/smousavi05?style=social)
![GitHub forks](https://img.shields.io/github/forks/smousavi05/EQTransformer?style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/smousavi05?style=social)


Important Note
--------------
.. image:: https://i.imgur.com/Ic8BA7C.png

This repository is related to the WhiteboxTools Python Frontend only. You can report issues to this repo if you have problems installing this Python package. If you encounter any tool functioning specific errors, please `open an issue`_ on Dr. John Lindsay's WhiteboxTools_ repo.  

**Links**

* Authors: S. Mostafa Mousavi
* GitHub repo: https://github.com/smousavi05/EQTransformer
* PyPI: https://pypi.org/project/EQTransformer/
* Anaconda: https://anaconda.org/smousavi05/eqtransformer
* Documentation:https://eqtransformer.readthedocs.io
* Free software: `MIT license`_

Description
-----------
**EQTransformer** is an AI-based earthquake signal detector and phase (P&S) picker based on a deep neural network with an attention mechanism. It has a hierarchical architecture specifically designed for earthquake signals. **EQTransformer** has been trained on global seismic data and can perform detection and arrival time picking simultaneously. In addition to the prediction probabilities, it also provides estimated model uncertainties.   
 
The ``EQTransformer`` python 3 package includes modules for downloading continuous seismic data, preprocessing, performing earthquake signal detection, and phase (P & S) picking using pre-trained models, building and testing new models, and performing a simple phase association. 


Reference
-----------
To cite **EQTransformer**, use the following reference To cite EQTransformer, use the following reference:

* Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L.Y., Beroza, G.C., "Earthquake Transformer: An Attentive Deep-learning Model for Simultaneous Earthquake Detection and Phase Picking ". Nature Communications, (2020).


Installation
------------
**EQTransformer** supports a variety of platforms, including Microsoft Windows, macOS, and Linux operating systems. Note that you will need to have Python 3.x (3.6 or 3.7) installed. The **EQTransformer** Python package can be installed using the following options:

#### Via Anaconda (recommended):


.. code:: python

  conda create -n eqt python=3.7

  Conda activate eqt

  conda config --add channels conda-forge

  conda install eqtransformer 
  

#### Via Pypi:

If you already have `Obspy <https://github.com/obspy/obspy/wiki/>`_ installed on your machine, you can get EQTransformer through Pypi:

.. code:: python

    pip install EQTransformer


#### From source:

The sources for EQTransformer can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code:: python

    git clone git://github.com/smousavi05/EQTransformer


Once you have a copy of the source, you can cd to EQTransformer directory and install it with:

.. code:: python

    python setup.py install


.. _Github repo: https://github.com/smousavi05/EQTransformer



If you have installed **EQTransformer** Python package before and want to upgrade to the latest version, you can use the following command:

.. code:: python

  pip install EQTransformer -U


Tutorials
---------

https://eqtransformer.readthedocs.io

or 

https://github.com/smousavi05/EQTransformer/tree/master/examples


Quick Example
=============

Tool names in the **EQTransformer** Python package can be called using the CamelCase convention (e.g. *LidarInfo*).

.. code:: python

    from EQTransformer.core.mseed_predictor import mseed_predictor
    
    mseed_predictor(input_dir= 'downloads_mseeds',   
             input_model='sampleData&Model/EqT1D8pre_048.h5',
             stations_json='station_list.json',
             output_dir='detections2',
             loss_weights=[0.02, 0.40, 0.58],          
             detection_threshold=0.3,                
             P_threshold=0.1,
             S_threshold=0.1, 
             number_of_plots=100,
             plot_mode = 'time_frequency',
             normalization_mode='std',
             batch_size=500,
             overlap = 0.3,
             gpuid=None,
             gpu_limit=None)



Contributing
------------

If you would like to contribute to the project as a developer, follow these instructions to get started:

1. Fork the EQTransformer project (https://github.com/smousavi05/EQTransformer)
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request


License
-------

The **EQTransformer** package is distributed under the `MIT license`_, a permissive open-source (free software) license.


Reporting Bugs
--------------
Report bugs at https://github.com/smousavi05/EQTransformer/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

