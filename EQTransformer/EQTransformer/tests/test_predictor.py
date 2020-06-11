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
    predictor(input_hdf5='test_data/guy_WHAR_aug_30s.hdf5',
              input_csv='test_data/guy_WHAR_aug_30s.csv',
              input_model='test_data/attentionCRED3br_14_T_data5_5CNN_Batch200_068.h5',
              output_name='test_predictor',
              output_probabilities=False,
              detection_threshold=0.20,                
              P_threshold=0.05,
              S_threshold=0.05, 
              number_of_plots=3,
              estimate_uncertainty=False, 
              number_of_sampling=5,
              batch_size=200,
              gpuid=None,
              gpu_limit=None)
    
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
