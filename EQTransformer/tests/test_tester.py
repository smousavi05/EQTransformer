#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 22:34:05 2020

@author: mostafamousavi
"""

from EQTransformer.core.tester import tester
import pytest
import glob
import os

def test_predictor():
    
    
    tester(input_hdf5='../sampleData&Model/100samples.hdf5',
       input_testset='test_trainer_outputs/test.npy',
       input_model='test_trainer_outputs/models/test_trainer_001.h5',
       output_name='test_tester',
       detection_threshold=0.20,                
       P_threshold=0.1,
       S_threshold=0.1, 
       number_of_plots=3,
       estimate_uncertainty=True, 
       number_of_sampling=2,
       input_dimention=(6000, 3),
       normalization_mode='std',
       mode='generator',
       batch_size=10,
       gpuid=None,
       gpu_limit=None)   
    
    dir_list = [ev for ev in os.listdir('.') if ev.split('_')[-1] == 'outputs']  
    
    if 'test_tester_outputs' in dir_list:
        successful = True
    else:
        successful = False        
    assert successful == True 
    
def test_report():
    report = glob.glob("test_tester_outputs/X_report.txt")
    assert len(report) == 1
    
def test_results():
    results = glob.glob("test_tester_outputs/X_test_results.csv")
    assert len(results) == 1
    
def test_plots():
    plots = glob.glob("test_tester_outputs/figures/*.png")
    assert len(plots) == 3