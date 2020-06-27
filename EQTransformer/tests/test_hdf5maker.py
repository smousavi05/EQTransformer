#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 22:34:05 2020

@author: mostafamousavi
"""

from EQTransformer.utils.hdf5_maker import preprocessor
import pytest
import glob
import os


def test_hdf5maker():
    
    preprocessor(mseed_dir='downloads_mseeds', 
             stations_json='station_list.json', 
             overlap=0.3,
             n_processor=2)

       
    dir_list = [ev for ev in os.listdir('.') if ev.split('_')[-1] == 'hdfs']  
        
    assert dir_list[0] == 'downloads_mseeds_processed_hdfs'
    
    
def test_outputs():
    
    output_files = glob.glob("downloads_mseeds_processed_hdfs/*")
    
    assert len(output_files) == 2
    
