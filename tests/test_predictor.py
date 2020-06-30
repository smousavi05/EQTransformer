#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 22:34:05 2020

@author: mostafamousavi
"""

from EQTransformer.core.predictor import predictor
import pytest
import glob
import os

def test_predictor():
        
    predictor(input_dir= 'downloads_mseeds_processed_hdfs',   
         input_model='../sampleData&Model/EqT1D8pre_048.h5',
         output_dir='detections',
         estimate_uncertainty=False, 
         output_probabilities=False,
         number_of_sampling=5,
         loss_weights=[0.02, 0.40, 0.58],          
         detection_threshold=0.30,                
         P_threshold=0.1,
         S_threshold=0.1, 
         number_of_plots=1000,
         plot_mode = 'time',
         batch_size=500,
         number_of_cpus=4,
         keepPS=False,
         spLimit=60) 

    
    dir_list = [ev for ev in os.listdir('.') if ev.split('_')[-1] == 'outputs']      
    assert dir_list[1] == 'test_predictor_outputs' 
    
def test_report():
    report = glob.glob("test_predictor_outputs/X_report.txt")
    assert len(report) == 1
    
def test_results():
    results = glob.glob("test_predictor_outputs/X_prediction_results.csv")
    assert len(results) == 1
    
def test_plots():
    plots = glob.glob("test_predictor_outputs/figures/*.png")
    assert len(plots) == 3
