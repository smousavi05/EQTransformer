#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 21:21:31 2019

@author: mostafamousavi

last update: 01/29/2021 
"""
import json
import time
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader
from obspy import UTCDateTime
import datetime
import os
import platform
from obspy.clients.fdsn.client import Client
import shutil
from multiprocessing.pool import ThreadPool
import multiprocessing
import numpy as np


def makeStationList(json_path,client_list, min_lat, max_lat, min_lon, max_lon, start_time, end_time, channel_list=[], filter_network=[], filter_station=[],**kwargs):


     """
    
    Uses fdsn to find available stations in a specific geographical location and time period.  

    Parameters
    ----------
    json_path: str
        Path of the json file that will be returned

    client_list: list
        List of client names e.g. ["IRIS", "SCEDC", "USGGS"].
                                
    min_lat: float
        Min latitude of the region.
        
    max_lat: float
        Max latitude of the region.
        
    min_lon: float
        Min longitude of the region.
        
    max_lon: float
        Max longitude of the region.
        
    start_time: str
        Start DateTime for the beginning of the period in "YYYY-MM-DDThh:mm:ss.f" format.
        
    end_time: str
        End DateTime for the beginning of the period in "YYYY-MM-DDThh:mm:ss.f" format.
        
    channel_list: str, default=[]
        A list containing the desired channel codes. Downloads will be limited to these channels based on priority. Defaults to [] --> all channels
        
    filter_network: str, default=[]
        A list containing the network codes that need to be avoided. 
        
    filter_station: str, default=[]
        A list containing the station names that need to be avoided.

    kwargs: 
        special symbol for passing Client.get_stations arguments

    Returns
    ----------
    stations_list.json: A dictionary containing information for the available stations.      
        
     """  
 
     station_list = {}
     for cl in client_list:
         inventory = Client(cl).get_stations(minlatitude=min_lat,
                                     maxlatitude=max_lat, 
                                     minlongitude=min_lon, 
                                     maxlongitude=max_lon, 
                                     starttime=UTCDateTime(start_time), 
                                     endtime=UTCDateTime(end_time), 
                                     level='channel',**kwargs)    

         for ev in inventory:
             net = ev.code
             if net not in filter_network:
                 for st in ev:
                     station = st.code
                     print(str(net)+"--"+str(station))
    
                     if station not in filter_station:

                         elv = st.elevation
                         lat = st.latitude
                         lon = st.longitude
                         new_chan = [ch.code for ch in st.channels]
                         if len(channel_list) > 0:
                             chan_priority=[ch[:2] for ch in channel_list]
        
                             for chnn in chan_priority:
                                 if chnn in [ch[:2] for ch in new_chan]:
                                     new_chan = [ch for ch in new_chan if ch[:2] == chnn]                     
    # =============================================================================
    #                      if ("BHZ" in new_chan) and ("HHZ" in new_chan):
    #                          new_chan = [ch for ch in new_chan if ch[:2] != "BH"]
    #                      if ("HHZ" in new_chan) and ("HNZ" in new_chan):
    #                          new_chan = [ch for ch in new_chan if ch[:2] != "HH"]
    #                          
    #                          if len(new_chan)>3 and len(new_chan)%3 != 0:
    #                              chan_type = [ch for ch in new_chan if ch[2] == 'Z']
    #                              chan_groups = []
    #                              for i, cht in enumerate(chan_type):
    #                                  chan_groups.append([ch for ch in new_chan if ch[:2] == cht[:2]])
    #                              new_chan2 = []
    #                              for chg in chan_groups:
    #                                  if len(chg) == 3:
    #                                      new_chan2.append(chg)
    #                              new_chan = new_chan2 
    # ============================================================================= 
                        
                         if len(new_chan) > 0 and (station not in station_list):
                             station_list[str(station)] ={"network": net,
                                                      "channels": list(set(new_chan)),
                                                      "coords": [lat, lon, elv]
                                                      }
     json_dir = os.path.dirname(json_path)
     if not os.path.exists(json_dir):
         os.makedirs(json_dir)
     with open(json_path, 'w') as fp:
         json.dump(station_list, fp)
         
         
def downloadMseeds(client_list, stations_json, output_dir, start_time, end_time, min_lat, max_lat, min_lon, max_lon, chunk_size, channel_list=[], n_processor=None):
    
    
    """
    
    Uses obspy mass downloader to get continuous waveforms from a specific client in miniseed format in variable chunk sizes. The minimum chunk size is 1 day. 
 
    Parameters
    ----------
    client_list: list
        List of client names e.g. ["IRIS", "SCEDC", "USGGS"].

    stations_json: dic,
        Station informations.
        
    output_dir: str
        Output directory.
                                
    min_lat: float
        Min latitude of the region.
        
    max_lat: float
        Max latitude of the region.
        
    min_lon: float
        Min longitude of the region.
        
    max_lon: float
        Max longitude of the region.
        
    start_time: str
        Start DateTime for the beginning of the period in "YYYY-MM-DDThh:mm:ss.f" format.
        
    end_time: str
        End DateTime for the beginning of the period in "YYYY-MM-DDThh:mm:ss.f" format.
        
    channel_list: str, default=[]
        A list containing the desired channel codes. Downloads will be limited to these channels based on priority. Defaults to [] --> all channels

    chunk_size: int
        Chunck size in day.
        
    n_processor: int, default=None
        Number of CPU processors for parallel downloading.

    Returns
    ----------

    output_name/station_name/*.mseed: Miniseed fiels for each station.      
 
    Warning
    ----------
    usage of multiprocessing and parallel downloads heavily depends on the client. It might cause missing some data for some network. Please test first for some short period and if It did miss some chunks of data for some channels then set n_processor=None to avoid parallel downloading.        
        
    """
     
         
    json_file = open(stations_json)
    station_dic = json.load(json_file)
    print(f"####### There are {len(station_dic)} stations in the list. #######")

    start_t = UTCDateTime(start_time)
    end_t = UTCDateTime(end_time)
    
    domain = RectangularDomain(minlatitude=min_lat, maxlatitude=max_lat, minlongitude=min_lon, maxlongitude=max_lon)
    mdl = MassDownloader(providers=client_list)

    bg=start_t 
 
    if n_processor==None:
        for st in station_dic:
            print(f'======= Working on {st} station.')
            _get_w(bg, st, station_dic, end_t, mdl, domain, output_dir, chunk_size, channel_list)
    else:        
        def process(st):
            print(f'======= Working on {st} station.')
            _get_w(bg, st, station_dic, end_t, mdl, domain, output_dir, chunk_size, channel_list)
            
        with ThreadPool(n_processor) as p:
            p.map(process, [ st for st in station_dic])      
        
       
    
    
def downloadSacs(client, stations_json, output_dir, start_time, end_time, patience, n_processor=None):
    
    """
    
    Uses obspy to get continuous waveforms from IRIS in sac format after preprocessing and in daily chunks. The difference to the mseed downloader is that this function removes the instrument response as it gets the data. 
 
    Parameters
    ----------
    client_list: list
        List of client names e.g. ["IRIS", "SCEDC", "USGGS"].

    stations_json: dic,
        Station informations.
 
    output_dir: str
        Output directory.
        
    start_time: str
        Start DateTime for the beginning of the period in "YYYY-MM-DDThh:mm:ss.f" format.
        
    end_time: str
        End DateTime for the beginning of the period in "YYYY-MM-DDThh:mm:ss.f" format.
                    
    patience: int
        The maximum number of days without data that the program allows continuing the downloading.
        
    chunk_size: int
        Chunck size in day.
        
      n_processor: int, default=None
        Number of CPU processors for parallel downloading. 
        

    Returns
    ----------
     
    output_name/station_name/*.SAC: SAC fiels for each station.      
 
    Warning
    ----------
    This function was not tested so you should be careful using it and make sure it gets the data.     
    
        
    """    
    


    if not n_processor:
        n_processor = multiprocessing.cpu_count()
        
    t_step = 86400   
    fr = open(stations_json, 'r'); 
    new_list = json.load(fr)
    print(f"####### There are {len(new_list)} stations in the list. #######")

    if platform.system() == 'Windows':
        if not os.path.exists(output_dir+"\\"):
            os.makedirs(output_dir+"\\")  
    else:        
        if not os.path.exists(output_dir+"/"):
            os.makedirs(output_dir+"/")  

    def process(station):          
        net = new_list[station]['network']
        
        if platform.system() == 'Windows':
            dirname = str(station)+"\\"
            if not os.path.exists(dirname):
                os.makedirs(dirname) 
        else:   
            dirname = str(station)+"/"
            if not os.path.exists(dirname):
                os.makedirs(dirname)  
                
        chans = new_list[station]['channels']  
        
        for chan in chans:
            print(f'======= Working on {station} station, {chan} channel.')
            unsucessful_downloads = []
            tstr = UTCDateTime(start_time)
            tend = UTCDateTime(start_time) + t_step   
            while tend <= UTCDateTime(end_time):  
                oo = _get_data(cel=client,
                              dirn=dirname,
                              net=net, 
                              station=station, 
                              chan=chan, 
                              starttime=tstr,
                              tend=tend,
                              count=0)
                unsucessful_downloads.append(oo)

                if sum(unsucessful_downloads) >= patience:
                    break
                    
                tstr = tend       
                tend = tend+t_step   
                  
        if len(os.listdir(dirname)) == 0: 
            os.rmdir(dirname)  
        else: 
            shutil.move(dirname, output_dir+"/"+dirname)
            
    with ThreadPool(n_processor) as p:
        p.map(process, new_list)             



def _get_w(bg, st, station_dic, end_t, mdl, domain, output_dir, n_days, channel_list):
    
    next_month=bg + datetime.timedelta(n_days)
    nt = station_dic[str(st)]['network'] 
    save_dir = os.path.join(output_dir, st)
    save_dir2 = os.path.join(output_dir+"xml", st)

    while next_month <= end_t:
        if len(channel_list) == 0:
            restrictions = Restrictions(starttime=bg,
                                        endtime=next_month,
                                        network=nt,
                                        station=st,
                                        reject_channels_with_gaps=False,
                                        minimum_length=0.0)
        else:
            restrictions = Restrictions(starttime=bg,
                                        endtime=next_month,
                                        network=nt,
                                        station=st,
                                        reject_channels_with_gaps=False,
                                        channel_priorities=channel_list,
                                        minimum_length=0.0)
        try:
            mdl.download(domain, 
                         restrictions,
                         mseed_storage = save_dir,
                         stationxml_storage = save_dir2)
            print(f"** done with --> {st} -- {nt} -- {str(bg).split('T')[0]}")                 

        except Exception:
            print(f"!! failed downloading --> {st} -- {nt} !")
            pass
        time.sleep(np.random.randint(25, 30))
        bg = next_month
        next_month = bg + datetime.timedelta(n_days)     




def _get_data(**kwargs):
    
    global out
    stio = kwargs['station']; cha = kwargs['chan']
    try:
        st = kwargs['cel'].get_waveforms(network=kwargs['net'],
                                   station=kwargs['station'],
                                   channel=kwargs['chan'],
                                   starttime=kwargs['starttime'],
                                   endtime=kwargs['tend'],
                                   location=False,
                                   attach_response=True)
        tt = str(kwargs['starttime']).split('T')[0]
        print(f"** --> got {stio} -- {cha} -- {tt}")                 
        st.merge(method=1, fill_value='interpolate')
      #  st.interpolate(sampling_rate=100)
        st[0].resample(100)
        st[0].data.dtype = 'int32'            
        st.detrend("demean")
        pre_filt = [0.8, 9.5, 40, 45] 
        st.remove_response(pre_filt=pre_filt,water_level=10,taper=True,taper_fraction=0.05)
      #  st.filter('bandpass',freqmin = 1.0, freqmax = 45, corners=2, zerophase=True) 
        st.write(filename=kwargs['dirn']+kwargs['net']+'.'+kwargs['station']+'..'+kwargs['chan']+"__"+str(kwargs['starttime']).split('T')[0].replace("-", "")+"T000000Z__"+str(kwargs['tend']).split('T')[0].replace("-", "")+"T000000Z.SAC",format="SAC")
        out = 0

    except:
        c = kwargs['count']
        print(f're-try downloading for {c} time!')
        kwargs['count'] += 1
        if kwargs['count'] <= 5:
            time.sleep(50)
            out = _get_data(cel=kwargs['cel'],
                     dirn=kwargs['dirn'],
                     net=kwargs['net'], 
                     station=kwargs['station'], 
                     chan=kwargs['chan'], 
                     starttime=kwargs['starttime'],
                     tend=kwargs['tend'],
                     count=kwargs['count'])
        else:
            print(f"!! didnt get ---> {stio} --- {cha}")
            out = 1            
    return out

    