  
![event](docs/source/figures/logo.png)            

# An AI-Based Earthquake Signal Detector and Phase Picker   

![PyPI](https://img.shields.io/pypi/v/EQTransformer?style=plastic)
![Conda](https://img.shields.io/conda/v/smousavi05/eqtransformer?style=plastic)
![Read the Docs](https://img.shields.io/readthedocs/eqtransformer?style=plastic)
![PyPI - License](https://img.shields.io/pypi/l/EQTransformer?style=plastic)
![Conda](https://img.shields.io/conda/dn/smousavi05/eqtransformer?style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/smousavi05/EQTransformer?style=plastic)
![Twitter Follow](https://img.shields.io/twitter/follow/smousavi05?style=social)
![GitHub followers](https://img.shields.io/github/followers/smousavi05?style=social)
![GitHub stars](https://img.shields.io/github/stars/smousavi05/EQTransformer?style=social)
![GitHub forks](https://img.shields.io/github/forks/smousavi05/EQTransformer?style=social)
 
--------------
## Description

**EQTransformer** is an AI-based earthquake signal detector and phase (P&S) picker based on a deep neural network with an attention mechanism. It has a hierarchical architecture specifically designed for earthquake signals. **EQTransformer** has been trained on global seismic data and can perform detection and arrival time picking simultaneously and efficiently. In addition to the prediction probabilities, it can also provide estimated model uncertainties.   
 
The **EQTransformer** python 3 package includes modules for downloading continuous seismic data, preprocessing, performing earthquake signal detection, and phase (P & S) picking using pre-trained models, building and testing new models, and performing a simple phase association. 

Developer: S. Mostafa Mousavi

* [Links](#Links) 
* [Reference](#Reference)                                
* [Installation](#Installation) 
* [Tutorials](#Tutorials)                   
* [A Quick Example](#A-Quick-Example)  
* [Test Set](#Test-Set)            
* [Contributing](#Contributing)                   
* [Reporting Bugs](#Reporting-Bugs)

-----------
## Links

* Documentation: https://rebrand.ly/EQT-documentations

* Paper:https://rdcu.be/b58li


-------------
## Reference

Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L, Y., and Beroza, G, C. Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nat Commun 11, 3952 (2020). https://doi.org/10.1038/s41467-020-17591-w

BibTeX:

    @article{mousavi2020earthquake,
        title={Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking},
        author={Mousavi, S Mostafa and Ellsworth, William L and Zhu, Weiqiang and Chuang, Lindsay Y and Beroza, Gregory C},
        journal={Nature Communications},
        volume={11},
        number={1},
        pages={1--12},
        year={2020},
        publisher={Nature Publishing Group}
    }

-----------------
## Installation

**EQTransformer** supports a variety of platforms, including macOS, Windows, and Linux operating systems. Note that you will need to have Python 3.x (3.6 or 3.7) installed. The **EQTransformer** Python package can be installed using the following options:

#### Via Anaconda (recommended):

    conda create -n eqt python=3.7

    conda activate eqt

    conda install -c smousavi05 eqtransformer 
    
##### Note: You may need to repeat executing the last line multiple time to succeed.  

#### Via PyPI:

If you already have `Obspy` installed on your machine, you can get **EQTransformer** through PyPI:

    pip install EQTransformer


#### From source:

The sources for **EQTransformer** can be downloaded from the `Github repo`.
##### Note: the gitgub version has been modified for Tensorflow 2.5.0

You can either clone the public repository:

    git clone git://github.com/smousavi05/EQTransformer
    
or (if you are working on Colab)

    pip install git+https://github.com/smousavi05/EQTransformer

Once you have a copy of the source, you can cd to **EQTransformer** directory and install it with:

    python setup.py install


If you have installed **EQTransformer** Python package before and want to upgrade to the latest version, you can use the following command:

    pip install EQTransformer -U

-------------
## Tutorials

See either:

https://rebrand.ly/EQT-documentations

and/or 

https://rebrand.ly/EQT-examples

Note: to run the notebook exampels, you may need to reinstall the jupyter on the same environment that **EQTransformer** has been installed.

**QuakeLabeler** (https://maihao14.github.io/QuakeLabeler/) is a software package that can be used to create labeled taining dataset for EQTransformer (i.e. STEAD format). 

**SeisBench** (https://github.com/seisbench/seisbench/)   (https://arxiv.org/abs/2111.00786) is an open source benchmarking package which makes it easy to apply pretrained EqT model, retraine it, fine tune it (transfer learning), or compare it with other models. 

-------------------
## A Quick Example

```python

    from EQTransformer.core.mseed_predictor import mseed_predictor
    
    mseed_predictor(input_dir='downloads_mseeds',   
                    input_model='ModelsAndSampleData/EqT_model.h5',
                    stations_json='station_list.json',
                    output_dir='detection_results',
                    detection_threshold=0.2,                
                    P_threshold=0.1,
                    S_threshold=0.1, 
                    number_of_plots=10,
                    plot_mode='time_frequency',
                    batch_size=500,
                    overlap=0.3)
```
-------------
## If you think that EqT does not detected all of the events in your experiment or produces too many false positives, please let us know. We are always interested to learn more about out of distribution cases to be able to improve our models.  

-------------
## Test Set

test.npy fine in the ModelsAndSampleData folder contains the trace names for the test set used in the paper. 
Based on these trace names you can retrieve our test data along with their labels from STEAD. Applying your model to these test traces you can directly compare the performance of your model to those in Tabels 1, 2, and 3 in the paper. 
The remaining traces in the STEAD were used for the training (85 %) and validation (5 %) respectively. 

---------------
## Contributing

If you would like to contribute to the project as a developer, follow these instructions to get started:

1. Fork the **EQTransformer** project (https://github.com/smousavi05/EQTransformer)
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

-----------
## License

The **EQTransformer** package is distributed under the `MIT license`, a permissive open-source (free software) license.

-----------------
## Reporting Bugs

Report bugs at https://github.com/smousavi05/EQTransformer/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

