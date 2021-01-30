Installation
=======================

**EQTransformer** is a Python 3.x package that uses libraries from `Tensorflow <https://www.tensorflow.org/>`_ and `Obspy <https://github.com/obspy/obspy/wiki/>`_.

Installation via conda (recommended)
------------------------------------
The following will download and install **EQTransformer** that supports a variety of platforms, including Windows, macOS, and Linux operating systems. Note that you will need to have Python 3.x (3.6 or 3.7) installed.

It is recommended that you use a Python virtual environment (e.g., conda) to test the **EQTransformer** package. Please follow the |conda user guide| to install conda if you do not have either a miniconda or anaconda installed on your machine. Once you have conda installed, you can use Terminal or an Anaconda Prompt to create a Python virtual environment. Check managing |anaconda environment| for more information.

.. code-block:: console

  conda create -n eqt python=3.7

  Conda activate eqt

  conda install -c smousavi05 eqtransformer


This will download and install **EQTransformer** and all required packages (including Tensorflow and Obspy) into your machine.
*Note:* Keep executing the last line if it did not succeed in the first try. 

.. |anaconda environment| raw:: html

    <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html" target="_blank">anaconda environment</a>

.. |conda user guide| raw:: html

    <a href="https://docs.anaconda.com/anaconda/user-guide/" target="_blank">conda user guide</a>

.. |conda-install| raw:: html

    <a href="https://docs.conda.io/en/latest/miniconda.html" target="_blank">conda-install</a>


Installation via PyPI
--------------------------
If you already have `Obspy <https://github.com/obspy/obspy/wiki/>`_ installed on your machine, you can get EQTransformer through PyPI:


.. code-block:: console

    pip install EQTransformer


If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/



Installation from Source
-------------------------

The sources for EQTransformer can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    git clone git://github.com/smousavi05/EQTransformer


Once you have a copy of the source, you can cd to EQTransformer directory and install it with:

.. code-block:: console

    python setup.py install


.. _Github repo: https://github.com/smousavi05/EQTransformer
