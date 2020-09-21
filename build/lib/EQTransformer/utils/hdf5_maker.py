#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 21:21:31 2019

@author: mostafamousavi

last update: 06-21-2020 

- downsampling using the interpolation function can cause false segmentaiton error. 
    This depend on your data and its sampling rate. If you kept getting this error when 
    using multiprocessors, try using only a single cpu. 
    
"""

from obspy import read
import os
from os import listdir
from os.path import join
import h5py
import numpy as np
import csv
#from tqdm import tqdm
import shutil
import json
import pandas as pd
from multiprocessing.pool import ThreadPool
import multiprocessing
import pickle
import faulthandler; faulthandler.enable()


def preprocessor(mseed_dir, stations_json, overlap=0.3, n_processor=None):
    
    
    """
    
    Performs preprocessing and partitions the continuous waveforms into 1-minute slices. 

    Parameters
    ----------
    mseed_dir: str
        A table containing trace names. 

    stations_json: str
        Path to a JSON file containing station information.        
        
    overlap: float, default=0.3
        If set, detection, and picking are performed in overlapping windows.
           
    n_processor: int, default=None 
        The number of CPU processors for parallel downloading.         

    Returns
    ----------
    mseed_dir_processed_hdfs/station.csv: Phase information for the associated events in hypoInverse format. 
    
    mseed_dir_processed_hdfs/station.hdf5: Containes all slices and preprocessed traces. 
    
    X_preprocessor_report.txt: A summary of processing performance. 
    
    time_tracks.pkl: Contain the time track of the continous data and its type.
       
    """  
 
    
    if not n_processor:
        n_processor = multiprocessing.cpu_count()
    
    json_file = open(stations_json)
    stations_ = json.load(json_file)
    
    save_dir = os.path.join(os.getcwd(), str(mseed_dir)+'_processed_hdfs')
    if os.path.isdir(save_dir):
        print(f' *** " {mseed_dir} " directory already exists!')
        inp = input(" * --> Do you want to creat a new empty folder? Type (Yes or y) ")
        if inp.lower() == "yes" or inp.lower() == "y":        
            shutil.rmtree(save_dir)  
    os.makedirs(save_dir)
              
    repfile = open("X_preprocessor_report.txt", 'w')
    station_list = [join(mseed_dir, ev) for ev in listdir(mseed_dir) if ev.split('/')[-1] != '.DS_Store'];
    
    data_track = dict()
    
    def process(station):
    # for station in station_list:
        output_name = station.split('/')[1]
        try:
            os.remove(output_name+'.hdf5')
            os.remove(output_name+".csv")
        except Exception:
            pass
        
        HDF = h5py.File(os.path.join(save_dir, output_name+'.hdf5'), 'a')
        HDF.create_group("data")
    
        csvfile = open(os.path.join(save_dir, output_name+".csv"), 'w')
        output_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(['trace_name', 'start_time'])
        csvfile.flush()   
    
        file_list = [join(station, ev) for ev in listdir(station) if ev.split('/')[-1] != '.DS_Store'];
        mon = [ev.split('__')[1]+'__'+ev.split('__')[2] for ev in file_list ];
        uni_list = list(set(mon))
        uni_list.sort()        
        tim_shift = int(60-(overlap*60))
        
        time_slots, comp_types = [], []
        
        print('============ Station {} has {} chunks of data.'.format(station.split('/')[1], len(uni_list)), flush=True)   
        count_chuncks=0; fln=0; c1=0; c2=0; c3=0; fl_counts=1; slide_estimates=[];
        
        for ct, month in enumerate(uni_list):
            matching = [s for s in file_list if month in s]
            
            if len(matching) == 3:  
                
                st1 = read(matching[0], debug_headers=True)
                org_samplingRate = st1[0].stats.sampling_rate
                
                for tr in st1:                   
                    time_slots.append((tr.stats.starttime, tr.stats.endtime))
                    comp_types.append(3)

                try:
                    st1.merge(fill_value=0) 
                except Exception:
                    st1=_resampling(st1)
                    st1.merge(fill_value=0)                     
                st1.detrend('demean') 
                count_chuncks += 1; c3 += 1
                print('  * '+station.split('/')[1]+' ('+str(count_chuncks)+') .. '+month.split('T')[0]+' --> '+month.split('__')[1].split('T')[0]+' .. 3 components .. sampling rate: '+str(org_samplingRate))  
                 
                st2 = read(matching[1], debug_headers=True) 
                try:
                    st2.merge(fill_value=0)                    
                except Exception:
                    st2=_resampling(st2)
                    st2.merge(fill_value=0)                    
                st2.detrend('demean')
    
                st3 = read(matching[2], debug_headers=True) 
                try:
                    st3.merge(fill_value=0)                     
                except Exception:
                    st3=_resampling(st3)
                    st3.merge(fill_value=0) 
                st3.detrend('demean')
                
                st1.append(st2[0])
                st1.append(st3[0])
                st1.filter('bandpass',freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
                st1.taper(max_percentage=0.001, type='cosine', max_length=2)
                if len([tr for tr in st1 if tr.stats.sampling_rate != 100.0]) != 0:
                    try:
                        st1.interpolate(100, method="linear")
                    except Exception:
                        st1=_resampling(st1)
                        
                                     
                longest = st1[0].stats.npts
                start_time = st1[0].stats.starttime
                end_time = st1[0].stats.endtime
                
                for tt in st1:
                    if tt.stats.npts > longest:
                        longest = tt.stats.npts
                        start_time = tt.stats.starttime
                        end_time = tt.stats.endtime
                    
                st1.trim(start_time, end_time, pad=True, fill_value=0)

                start_time = st1[0].stats.starttime
                end_time = st1[0].stats.endtime  
                slide_estimates.append((end_time - start_time)//tim_shift)                
                fl_counts += 1 
                
                chanL = [st1[0].stats.channel[-1], st1[1].stats.channel[-1], st1[2].stats.channel[-1]]
                next_slice = start_time+60               
                while next_slice <= end_time:
                    w = st1.slice(start_time, next_slice) 
                    npz_data = np.zeros([6000,3])
                                        
                    npz_data[:,2] = w[chanL.index('Z')].data[:6000]
                    try: 
                        npz_data[:,0] = w[chanL.index('E')].data[:6000]
                    except Exception:
                        npz_data[:,0] = w[chanL.index('1')].data[:6000]
                    try: 
                        npz_data[:,1] = w[chanL.index('N')].data[:6000]
                    except Exception:
                        npz_data[:,1] = w[chanL.index('2')].data[:6000]                        
                                     
                    tr_name = st1[0].stats.station+'_'+st1[0].stats.network+'_'+st1[0].stats.channel[:2]+'_'+str(start_time)
                    HDF = h5py.File(os.path.join(save_dir,output_name+'.hdf5'), 'r')
                    dsF = HDF.create_dataset('data/'+tr_name, npz_data.shape, data = npz_data, dtype= np.float32)        
                       
                    dsF.attrs["trace_name"] = tr_name 
                    dsF.attrs["receiver_code"] = station.split('/')[1]
                    dsF.attrs["network_code"] = stations_[station.split('/')[1]]['network']
                    dsF.attrs["receiver_latitude"] = stations_[station.split('/')[1]]['coords'][0]
                    dsF.attrs["receiver_longitude"] = stations_[station.split('/')[1]]['coords'][1]
                    dsF.attrs["receiver_elevation_m"] = stations_[station.split('/')[1]]['coords'][2]    
                    start_time_str = str(start_time)   
                    start_time_str = start_time_str.replace('T', ' ')                 
                    start_time_str = start_time_str.replace('Z', '')          
                    dsF.attrs['trace_start_time'] = start_time_str
                    HDF.flush()
                    output_writer.writerow([str(tr_name), start_time_str])  
                    csvfile.flush()
                    fln += 1            
            
                    start_time = start_time+tim_shift
                    next_slice = next_slice+tim_shift 
  
            if len(matching) == 1:  
                 count_chuncks += 1; c1 += 1
                
                 st1 = read(matching[0], debug_headers=True)
                 org_samplingRate = st1[0].stats.sampling_rate

                 for tr in st1:                   
                     time_slots.append((tr.stats.starttime, tr.stats.endtime))
                     comp_types.append(1)
                 try:
                     st1.merge(fill_value=0) 
                 except Exception:
                     st1=_resampling(st1)
                     st1.merge(fill_value=0)                 
                 st1.detrend('demean')     
                 
                 print('  * '+station.split('/')[1]+' ('+str(count_chuncks)+') .. '+month.split('T')[0]+' --> '+month.split('__')[1].split('T')[0]+' .. 1 components .. sampling rate: '+str(org_samplingRate)) 
                 
                 st1.filter('bandpass',freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
                 st1.taper(max_percentage=0.001, type='cosine', max_length=2)
                 if len([tr for tr in st1 if tr.stats.sampling_rate != 100.0]) != 0:
                     try:
                         st1.interpolate(100, method="linear")
                     except Exception:
                         st1=_resampling(st1) 
                         
                 chan = st1[0].stats.channel
                 start_time = st1[0].stats.starttime
                 end_time = st1[0].stats.endtime
                 slide_estimates.append((end_time - start_time)//tim_shift)
                 fl_counts += 1    

                 next_slice = start_time+60

                 while next_slice <= end_time:
                     w = st1.slice(start_time, next_slice)                    
                     npz_data = np.zeros([6000,3])
                     if chan[-1] == 'Z':
                         npz_data[:,2] = w[0].data[:6000]
                     if chan[-1] == 'E' or  chan[-1] == '1':
                         npz_data[:,0] = w[0].data[:6000]
                     if chan[-1] == 'N' or  chan[-1] == '2':
                         npz_data[:,1] = w[0].data[:6000]
                    
                     tr_name = st1[0].stats.station+'_'+st1[0].stats.network+'_'+st1[0].stats.channel[:2]+'_'+str(start_time)
                     HDF = h5py.File(os.path.join(save_dir,output_name+'.hdf5'), 'r')
                     dsF = HDF.create_dataset('data/'+tr_name, npz_data.shape, data = npz_data, dtype= np.float32)        
                     dsF.attrs["trace_name"] = tr_name 
                     dsF.attrs["receiver_code"] = station.split('/')[1]
                     dsF.attrs["network_code"] = stations_[station.split('/')[1]]['network']
                     dsF.attrs["receiver_latitude"] = stations_[station.split('/')[1]]['coords'][0]
                     dsF.attrs["receiver_longitude"] = stations_[station.split('/')[1]]['coords'][1]
                     dsF.attrs["receiver_elevation_m"] = stations_[station.split('/')[1]]['coords'][2]    
                     start_time_str = str(start_time)   
                     start_time_str = start_time_str.replace('T', ' ')                 
                     start_time_str = start_time_str.replace('Z', '')          
                     dsF.attrs['trace_start_time'] = start_time_str
                     HDF.flush()
                     output_writer.writerow([str(tr_name), start_time_str])  
                     csvfile.flush()
                     fln += 1            

                     start_time = start_time+tim_shift
                     next_slice = next_slice+tim_shift                
                
            if len(matching) == 2:  
                count_chuncks += 1; c2 += 1                
                st1 = read(matching[0], debug_headers=True)
                org_samplingRate = st1[0].stats.sampling_rate

                for tr in st1:                   
                    time_slots.append((tr.stats.starttime, tr.stats.endtime))
                    comp_types.append(2)

                try:
                    st1.merge(fill_value=0) 
                except Exception:
                    st1=_resampling(st1)
                    st1.merge(fill_value=0)  
                st1.detrend('demean')  
                
                org_samplingRate = st1[0].stats.sampling_rate
                print('  * '+station.split('/')[1]+' ('+str(count_chuncks)+') .. '+month.split('T')[0]+' --> '+month.split('__')[1].split('T')[0]+' .. 2 components .. sampling rate: '+str(org_samplingRate)) 
                 
                st2 = read(matching[1], debug_headers=True)  
                try:
                    st2.merge(fill_value=0) 
                except Exception:
                    st2=_resampling(st1)
                    st2.merge(fill_value=0)                 
                st2.detrend('demean')
    
                st1.append(st2[0])
                st1.filter('bandpass',freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
                st1.taper(max_percentage=0.001, type='cosine', max_length=2)
                if len([tr for tr in st1 if tr.stats.sampling_rate != 100.0]) != 0:
                    try:
                        st1.interpolate(100, method="linear")
                    except Exception:
                        st1=_resampling(st1)   
                        
                longest = st1[0].stats.npts
                start_time = st1[0].stats.starttime
                end_time = st1[0].stats.endtime
                
                for tt in st1:
                    if tt.stats.npts > longest:
                        longest = tt.stats.npts
                        start_time = tt.stats.starttime
                        end_time = tt.stats.endtime               
                
                st1.trim(start_time, end_time, pad=True, fill_value=0)

                start_time = st1[0].stats.starttime
                end_time = st1[0].stats.endtime
                slide_estimates.append((end_time - start_time)//tim_shift)
                
                chan1 = st1[0].stats.channel
                chan2 = st1[1].stats.channel
                fl_counts += 1  
                
                next_slice = start_time+60

                while next_slice <= end_time:
                    w = st1.slice(start_time, next_slice)                     
                    npz_data = np.zeros([6000,3])
                    if chan1[-1] == 'Z':
                        npz_data[:,2] = w[0].data[:6000]
                    elif chan1[-1] == 'E' or  chan1[-1] == '1':
                        npz_data[:,0] = w[0].data[:6000]
                    elif chan1[-1] == 'N' or  chan1[-1] == '2':
                        npz_data[:,1] = w[0].data[:6000]

                    if chan2[-1] == 'Z':
                        npz_data[:,2] = w[1].data[:6000]
                    elif chan2[-1] == 'E' or  chan2[-1] == '1':
                        npz_data[:,0] = w[1].data[:6000]
                    elif chan2[-1] == 'N' or  chan2[-1] == '2':
                        npz_data[:,1] = w[1].data[:6000]
                    
                    tr_name = st1[0].stats.station+'_'+st1[0].stats.network+'_'+st1[0].stats.channel[:2]+'_'+str(start_time)
                    HDF = h5py.File(os.path.join(save_dir,output_name+'.hdf5'), 'r')
                    dsF = HDF.create_dataset('data/'+tr_name, npz_data.shape, data = npz_data, dtype= np.float32)        
                       
                    dsF.attrs["trace_name"] = tr_name 
                    dsF.attrs["receiver_code"] = station.split('/')[1]
                    dsF.attrs["network_code"] = stations_[station.split('/')[1]]['network']
                    dsF.attrs["receiver_latitude"] = stations_[station.split('/')[1]]['coords'][0]
                    dsF.attrs["receiver_longitude"] = stations_[station.split('/')[1]]['coords'][1]
                    dsF.attrs["receiver_elevation_m"] = stations_[station.split('/')[1]]['coords'][2]    
                    start_time_str = str(start_time)   
                    start_time_str = start_time_str.replace('T', ' ')                 
                    start_time_str = start_time_str.replace('Z', '')          
                    dsF.attrs['trace_start_time'] = start_time_str
                    HDF.flush()
                    output_writer.writerow([str(tr_name), start_time_str])  
                    csvfile.flush()
                    fln += 1            
            
                    start_time = start_time+tim_shift
                    next_slice = next_slice+tim_shift 
                    
            st1, st2, st3 = None, None, None
                
        HDF.close() 
        
        dd = pd.read_csv(os.path.join(save_dir, output_name+".csv"))
                
        
        assert count_chuncks == len(uni_list)  
        assert sum(slide_estimates)-(fln/100) <= len(dd) <= sum(slide_estimates)+10
        data_track[output_name]=[time_slots, comp_types]
        print(f" Station {output_name} had {len(uni_list)} chuncks of data") 
        print(f"{len(dd)} slices were written, {sum(slide_estimates)} were expected.")
        print(f"Number of 1-components: {c1}. Number of 2-components: {c2}. Number of 3-components: {c3}.")
        try:
            print(f"Original samplieng rate: {org_samplingRate}.") 
            repfile.write(f' Station {output_name} had {len(uni_list)} chuncks of data, {len(dd)} slices were written, {int(sum(slide_estimates))} were expected. Number of 1-components: {c1}, Number of 2-components: {c2}, number of 3-components: {c3}, original samplieng rate: {org_samplingRate}\n')
        except Exception:
            pass
    with ThreadPool(n_processor) as p:
        p.map(process, station_list) 
    with open('time_tracks.pkl', 'wb') as f:
        pickle.dump(data_track, f, pickle.HIGHEST_PROTOCOL)

def _resampling(st):
    need_resampling = [tr for tr in st if tr.stats.sampling_rate != 100.0]
    if len(need_resampling) > 0:
       # print('resampling ...', flush=True)    
        for indx, tr in enumerate(need_resampling):
            if tr.stats.delta < 0.01:
                tr.filter('lowpass',freq=45,zerophase=True)
            tr.resample(100)
            tr.stats.sampling_rate = 100
            tr.stats.delta = 0.01
            tr.data.dtype = 'int32'
            st.remove(tr)                    
            st.append(tr)    
             
    return st