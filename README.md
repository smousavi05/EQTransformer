  
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

Developer: S. Mostafa Mousavi (smousavi05@gmail.com)

* [Links](#Links) 
* [Installation](#Installation) 
* [Tutorials](#Tutorials)  
* [Related Software Packages](#Related-Software-Packages)                 
* [A Quick Example](#A-Quick-Example)  
* [Reference](#Reference)                                
* [Test Set](#Test-Set)            
* [Contributing](#Contributing)                   
* [Reporting Bugs](#Reporting-Bugs)
* [Studies That Used EqT](#Some-of-the-studies-in-which-EqT-has-been-used)

-----------
## Links

* Documentation: https://rebrand.ly/EQT-documentations

* Paper:https://rdcu.be/b58li


-----------------
## Installation

**EQTransformer** supports a variety of platforms, including macOS, Windows, and Linux operating systems. Note that you will need to have Python 3.x (3.6 or 3.7) installed. The **EQTransformer** Python package can be installed using the following options:

#### Via Anaconda (recommended):

    conda create -n eqt python=3.7

    conda activate eqt

    conda install -c smousavi05 eqtransformer 
    
##### Note: sometimes you need to keep repeating executing the last line multiple time to succeed.  

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
    
##### To nstall EqT on M1 laptop with python>=3.9 from the source (GitHub) code by changing tensorflow to tensorflow-maco in the setup.py and follow the these steps:

      conda create -n eqt python=3.10
      conda activate eqt
      conda install -c apple tensorflow-deps
      conda install obspy jupyter pandas
      pip install tensorflow-macos
      python3 setup.py install

-------------
## Tutorials

See either:

https://rebrand.ly/EQT-documentations

and/or 

https://rebrand.ly/EQT-examples

### Important Notes:
There is not much difference between the two models (i.e. original and conservative models) regarding the network architecture (only 1 or 2 layers). The main difference comes from the training procedure and the hyperparameters used for data augmentation during the training. 

***Original model*** (the one in the paper) has been optimized to minimize the false negative rate (maximize the number of detected events). This is a good choice when high false positive rates is not an issue in your application. For instance, in event location workfolow most of the false positives will automatically be removed during the association, location, and - especially - relocation procedure. You can use the original model with higher threshold values (~ 0.3 for P and S and 0.5 for detection).

***Conservative Model***, on the other hand, has been optimized to minimize the false positive rate (maximum number of valid detections). This model is useful when your application can not tolerate a high false positive rate. For example, in travel time tomography or using the detected events by EqT as the templates for template matching. You should use this model with a much lower threshold levels (0.03 for P and S). 

***Suggestions for Optimal Catalog Building***:
1) use the original model for detection and picking and have a cross-correlation-based event relocation at the end to clean up false positives.
2) use detected events by the conservative model as template events and then perform a complementary template-matching to detect the rest of events. See this paper (https://academic.oup.com/gji/advance-article-abstract/doi/10.1093/gji/ggac487/6881721) as an example.

***If you feel some larger events are missed while smaller ones are detected***:
Use larger overlapping value (e.g. 0.9) for moving windows.

Note: to run the notebook exampels, you may need to reinstall the jupyter on the same environment that **EQTransformer** has been installed.

-------------------
## Related Software Packages:

***Blocky Earthquake Transformer*** (https://github.com/maihao14/BlocklyEQTransformer) is a user-interface no-code-based platform that makes it easy to fine-tune EqT for specific data/regions. It also provides a user-friendly interface to create your own EqT model and train it without dealing with coding or knowing in-depth ML. 

***Siamese Earthquake Transformer*** (https://github.com/MrXiaoXiao/SiameseEarthquakeTransformer) performes a secondary template-matching-type post-processing step (by using the learned features in EqT layers as the templates and measure the similarities using a Siamese neural network instead of cross-correlation) to reduce the false-negative rate of the EqT retrieving previously missed phase picks in low SNR seismograms.

***OBSTransformer*** (https://github.com/alirezaniki/OBSTransformer)) is a transfer-learned seismic phase picker for Ocean Bottom Seismometer (OBS) data adopted from the EqTransformer model. OBSTransformer has been trained on an auto-labeled tectonically inclusive OBS dataset comprising ~36k earthquake and 25k noise samples.

***QuakeLabeler*** (https://maihao14.github.io/QuakeLabeler/) is a software package that can be used to create labeled training dataset for EQTransformer (i.e. STEAD format). 

***SeisBench*** (https://github.com/seisbench/seisbench/) is an open source benchmarking package with Pytorch implementaion of EgT that makes it easy to either apply pre-trained EqT model, retrain it, or compare it with other models. 

***MALMI*** (https://github.com/speedshi/MALMI/) is an earthquake monitoring pipline, i.e. picking and event location determination, that uses EqT for event detection and phase picking.  

***easyQuake*** (https://github.com/jakewalter/easyQuake) is an earthquake monitoring pipline, i.e. detection, picking, association, location, and magnitude determination, that icludes EqT and other DL-pickers for event detection and phase picking.  


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


-----------
## License

The **EQTransformer** package is distributed under the `MIT license`, a permissive open-source (free software) license.

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

-----------------
## Reporting Bugs
#### Please see https://github.com/smousavi05/EQTransformer/issues?q=is%3Aissue+is%3Aclosed for a list of issues/bugs that have been already reported/fixed before filing a new bug report.

Report bugs at https://github.com/smousavi05/EQTransformer/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

-----------------
## Some of the studies in which EqT has been used:

* di Uccio, F. S., Scala, A., Festa, G., Picozzi, M., & Beroza, G. C. (2022). Comparing and integrating artificial intelligence and similarity search detection techniques: application to seismic sequences in Southern Italy. Authorea Preprints.

* Gong, J., & Fan, W. (2022). Seismicity, fault architecture, and slip mode of the westernmost Gofar transform fault. Journal of Geophysical Research: Solid Earth, 127(11), e2022JB024918.

* Shiddiqi, H. A., Ottemöller, L., Rondenay, S., Custódio, S., Halpaap, F., & Gahalaut, V. K. (2023). Comparison of Earthquake Clusters in a Stable Continental Region: A Case Study from Nordland, Northern Norway. Seismological Research Letters.

* Chin, S. J., Sutherland, R., Savage, M. K., Townend, J., Collot, J., Pelletier, B., ... & Illsley‐Kemp, F. (2022). Earthquakes and Seismic Hazard in Southern New Caledonia, Southwest Pacific. Journal of Geophysical Research: Solid Earth, e2022JB024207.

* Miller, M. S., Pickle, R., Murdie, R., Yuan, H., Allen, T. I., Gessner, K., ... & Whitney, J. (2023). Southwest Australia Seismic Network (SWAN): Recording Earthquakes in Australia’s Most Active Seismic Zone. Seismological Research Letters.

* Zhang, Z., Deng, Y., Qiu, H., Peng, Z., & Liu‐Zeng, J. (2022). High‐Resolution Imaging of Fault Zone Structure Along the Creeping Section of the Haiyuan Fault, NE Tibet, From Data Recorded by Dense Seismic Arrays. Journal of Geophysical Research: Solid Earth, 127(9), e2022JB024468.

* Bannister, S., Bertrand, E. A., Heimann, S., Bourguignon, S., Asher, C., Shanks, J., & Harvison, A. (2022). Imaging sub-caldera structure with local seismicity, Okataina Volcanic Centre, Taupo Volcanic Zone, using double-difference seismic tomography. Journal of Volcanology and Geothermal Research, 431, 107653.

* Michailos, K., Carpenter, N. S., & Hetényi, G. (2021). Spatio-temporal evolution of intermediate-depth seismicity beneath the Himalayas: Implications for metamorphism and tectonics. Frontiers in Earth Science, 859.

* Gong, J., Fan, W., & Parnell‐Turner, R. (2022). Microseismicity Indicates Atypical Small‐Scale Plate Rotation at the Quebrada Transform Fault System, East Pacific Rise. Geophysical Research Letters, 49(3), e2021GL097000.

* Jiang, C., Zhang, P., White, M. C., Pickle, R., & Miller, M. S. (2022). A Detailed Earthquake Catalog for Banda Arc–Australian Plate Collision Zone Using Machine‐Learning Phase Picker and an Automated Workflow. The Seismic Record, 2(1), 1-10.

* Chuang, L. Y., Peng, Z., Lei, X., Wang, B., Liu, J., Zhai, Q., & Tu, H. (2023). Foreshocks of the 2010 Mw 6.7 Yushu, China Earthquake Occurred Near an Extensional Step‐Over. Journal of Geophysical Research: Solid Earth, 128(1), e2022JB025176.

* Kemna, K. B., Roth, M. P., Wache, R. M., Harrington, R. M., & Liu, Y. Small magnitude events highlight the correlation between hydraulic fracturing injection parameters, geological factors, and earthquake occurrence. Geophysical Research Letters, e2022GL099995.

* Cianetti, S., Bruni, R., Gaviano, S., Keir, D., Piccinini, D., Saccorotti, G., & Giunchi, C. (2021). Comparison of deep learning techniques for the investigation of a seismic sequence: An application to the 2019, Mw 4.5 Mugello (Italy) earthquake. Journal of Geophysical Research: Solid Earth, 126(12), e2021JB023405.

* van der Laat, L., Baldares, R. J., Chaves, E. J., & Meneses, E. (2021, November). OKSP: A Novel Deep Learning Automatic Event Detection Pipeline for Seismic Monitoring in Costa Rica. In 2021 IEEE 3rd International Conference on BioInspired Processing (BIP) (pp. 1-6). IEEE.

* Caudron, C., Aoki, Y., Lecocq, T., De Plaen, R., Soubestre, J., Mordret, A., ... & Terakawa, T. (2022). Hidden pressurized fluids prior to the 2014 phreatic eruption at Mt Ontake. Nature communications, 13(1), 1-9.

* WEN, X., SHEN, X., & ZHOU, Q. (2022). Study on the characters of the aftershocks of Beiliu 5.2 earthquake using machine learning method and dense nodal seismic array. Chinese Journal of Geophysics, 65(9), 3297-3308.

* Sheng, Y., Pepin, K. S., & Ellsworth, W. L. (2022). On the Depth of Earthquakes in the Delaware Basin: A Case Study along the Reeves–Pecos County Line. The Seismic Record, 2(1), 29-37.

* Walter, J. I., Ogwari, P., Thiel, A., Ferrer, F., & Woelfel, I. (2021). easyQuake: Putting machine learning to work for your regional seismic network or local earthquake study. Seismological Research Letters, 92(1), 555-563.

* Shi, P., Grigoli, F., Lanza, F., Beroza, G. C., Scarabello, L., & Wiemer, S. (2022). MALMI: An Automated Earthquake Detection and Location Workflow Based on Machine Learning and Waveform Migration. Seismological Research Letters.


