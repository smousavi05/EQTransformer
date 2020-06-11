#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 22:34:05 2020

@author: mostafamousavi
"""

from EQTransformer.utils.downloader import download
import pytest
import glob
import os


def test_downloader():
    download(stations_json='test_data/station_list2.json', 
               output_dir="downloaded_mseeds", 
               start_time="2018-01-01T00:00:00.0", 
               end_time="2018-02-01T00:00:00.0", 
               minlat=35.0, 
               maxlat=37.0, 
               minlon=-91.0, 
               maxlon=-89.0)
       
    dir_list = [ev for ev in os.listdir('.') if ev.split('_')[-1] == 'mseeds']  
    if 'downloaded_mseeds'  in dir_list:
        successful = True
    else:
        successful = False 
        
    assert successful == True
    
    
def test_mseeds():
    
    mseeds = glob.glob("downloaded_mseeds/ARPT/*.mseed")
    
    assert len(mseeds) == 3
