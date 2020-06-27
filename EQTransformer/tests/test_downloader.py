#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 22:34:05 2020

@author: mostafamousavi
"""

from EQTransformer.utils.downloader import downloadMseeds, makeStationList, downloadSacs
import pytest
import glob
import os



def test_downloader():
    
    makeStationList(client_list=["SCEDC"],  
                  min_lat=35.50,
                  max_lat=35.60,
                  min_lon=-117.80, 
                  max_lon=-117.40,                      
                  start_time="2019-09-01 00:00:00.00", 
                  end_time="2019-09-03 00:00:00.00",
                  channel_list=["HH[ZNE]", "HH[Z21]", "BH[ZNE]", "EH[ZNE]", "SH[ZNE]", "HN[ZNE]", "HN[Z21]", "DP[ZNE]"],
                  filter_network=["SY"],
                  filter_station=[])
    
    
    downloadMseeds(client_list=["SCEDC", "IRIS"], 
              stations_json='station_list.json', 
              output_dir="downloads_mseeds", 
              start_time="2019-09-01 00:00:00.00", 
              end_time="2019-09-02 00:00:00.00", 
              min_lat=35.50,
              max_lat=35.60,
              min_lon=-117.80, 
              max_lon=-117.40,
              chunck_size=1,
              channel_list=[],
              n_processor=2)
    
        
    dir_list = [ev for ev in os.listdir('.')]  
    if ('downloads_mseeds' in dir_list) and ('station_list.json' in dir_list):
        successful = True
    else:
        successful = False 
        
    assert successful == True
    
    
def test_mseeds():
    
    mseeds = glob.glob("downloads_mseeds/CA06/*.mseed")
    
    assert len(mseeds) > 0
