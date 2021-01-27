"""
Created on Wed Jul 24 19:16:51 2019

@author: mostafamousavi
last update: 06/05/2020

"""
import pandas as pd
from os import listdir
import platform
import json
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
import matplotlib.cm
import pickle
import datetime as dt
from matplotlib.dates import  HOURLY, DateFormatter, rrulewrapper, RRuleLocator
import matplotlib.font_manager as font_manager
from obspy import read
from obspy import UTCDateTime




def plot_helicorder(input_mseed, input_csv=None, save_plot=False):
    
    """
    
    Plots an stream object overlied by detection times. 

    Parameters
    ----------       
    input_mseed: str
        Path to the miniseed files for day long data.  
        
    input_csv: str, default=None
        Path to the "X_prediction_results.csv" file associated with the miniseed file.                   
        
    save_plot: str, default=False
        If set to True the generated plot will be saved with the name of miniseed file. 
                          

    Returns
    ----------       
    miniseed_name.png: fig
       
    """  
          
    event_list=[]
    def _date_convertor(r):            
        r.replace(' ', 'T')+'Z'
        new_t = UTCDateTime(r)
        return new_t
         
    st = read(input_mseed)
    st.filter("highpass", freq=0.1, corners=2)

    if input_csv:
        detlist = pd.read_csv(input_csv)       
        detlist['event_start_time'] = detlist['event_start_time'].apply(lambda row : _date_convertor(row)) 
        detlist = detlist[(detlist.event_start_time > st[0].stats['starttime']) & (detlist.event_start_time < st[0].stats['endtime'])]
        ev_list = detlist['event_end_time'].to_list()
        for ev in ev_list:
            event_list.append({"time": UTCDateTime(ev)})
    if save_plot:    
        if platform.system() == 'Windows':
            st[0].plot(type='dayplot', color=['k'],interval=60, events=event_list, outfile=input_mseed.split("\\")[-1].split('.mseed')[0]+'.png')
            print('saved the plot as '+input_mseed.split("\\")[-1].split('.mseed')[0]+'.png')
        else:    
            st[0].plot(type='dayplot', color=['k'],interval=60, events=event_list, outfile=input_mseed.split("/")[-1].split('.mseed')[0]+'.png')
            print('saved the plot as '+input_mseed.split("/")[-1].split('.mseed')[0]+'.png')
    else:    
        st[0].plot(type='dayplot', color=['k'],interval=60, events=event_list)





def plot_detections(input_dir, input_json, plot_type=None, time_window=60, marker_size=6):
    
    
     """
    
     Uses fdsn to find availave stations in a specific geographical location and time period. 

    Parameters
    ----------       
    input_dir: str
        Path to the directory containing detection results.
         
    input_json: str
        Json file containing station information.
         
    plot_type: str, default=None
        Type of plot, 'station_map', 'hist'. 
           
    time_window: int, default=60 
        Time window for histogram plot in minutes. 
        

    Returns
    ----------   
    station_output.png: fig
        
    station_map.png: fig
       
     """  

     if platform.system() == 'Windows':
         station_list = [ev for ev in listdir(input_dir) if ev.split("\\")[-1] != '.DS_Store'];
     else:
         station_list = [ev for ev in listdir(input_dir) if ev.split("/")[-1] != '.DS_Store'];

     station_list = sorted(set(station_list))
    
    
     json_file = open(input_json)
     stations_ = json.load(json_file)
    
     detection_list = {}
     for st in station_list: 
         if platform.system() == 'Windows':
             df_mulistaition = pd.read_csv(input_dir+"\\"+st+'"\\"X_prediction_results.csv') 
         else:
             df_mulistaition = pd.read_csv(input_dir+"/"+st+"/X_prediction_results.csv") 
             
         detection_list[st.split("_")[0]]=[stations_[st.split("_")[0]]['coords'][1],stations_[st.split("_")[0]]['coords'][0],len(df_mulistaition)]
    
     ln2=[]; lt2=[]; detections=[]
     for stations, L in detection_list.items():
         ln2.append(L[0])
         lt2.append(L[1])
         detections.append(L[2])
        
     if plot_type == 'station_map':
         
         plt.figure(constrained_layout=True) 
         plt.scatter(ln2, lt2, s=marker_size, marker="^", c=detections)
         plt.xticks(rotation=45)
         c = plt.colorbar(orientation='vertical')
         c.set_label("Number of Detections")
        
         for stations, L in detection_list.items(): 
             plt.text(L[0], L[1], stations, fontsize=marker_size//4)
            
         plt.title(str(len(detection_list))+' Stations')
         plt.savefig('station_map.png', dpi=300)  
         plt.tight_layout()   
         plt.show()
         
     elif plot_type == 'hist':
         for st in station_list: 
             if platform.system() == 'Windows':
                 df = pd.read_csv(input_dir+"\\"+st+'"\\"X_prediction_results.csv')     
             else:    
                 df = pd.read_csv(input_dir+"/"+st+"/X_prediction_results.csv")
                 
             df['event_start_time'] = df['event_start_time'].apply(lambda row : _date_convertor(row)) 

             plt.figure(constrained_layout=True)
             df.set_index('event_start_time', drop=False, inplace=True)
             df = df['event_start_time'].groupby(pd.Grouper(freq=str(time_window)+'Min')).count()
             ax = df.plot(kind='bar', color='slateblue')
             ticklabels = df.index.strftime('%D:%H')
             ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(ticklabels))
             ax.set_ylabel('Event Counts')
             plt.title(st)
             plt.savefig(st+'.png', dpi=300)
             plt.show()
     else:
         print('Please define the plot type!')
          
         
         


def plot_data_chart(time_tracks, time_interval):
    
    """
    
    Uses fdsn to find availave stations in a specific geographical location and time period. 

    Parameters
    ----------      
    time_tracks: pkl
        Pickel file outputed by preprocessor or mseed_predictor.     
    
    time_interval: int 
        Time interval in hours for tick spaces in xaxes. 
               

    Returns
    ----------      
    data_chart.png: fig
       
    """      
     
     
    with open(time_tracks, 'rb') as f:
        time_tracks = pickle.load(f)
        
    def create_date(date_time):
     
        date = dt.datetime(int(str(date_time).split('T')[0].split('-')[0]),
                           int(str(date_time).split('T')[0].split('-')[1]), 
                           int(str(date_time).split('T')[0].split('-')[2]),
                           int(str(date_time).split('T')[1].split(':')[0]),
                           int(str(date_time).split('T')[1].split(':')[1])
                           )   
        mdate = matplotlib.dates.date2num(date)
         
        return mdate

    ylabels, customDates, task_dates = [], [], {}
    for st, tracks in time_tracks.items():
        ylabels.append(st)
        time_slots, comp_types = [], []
        for times in tracks[0]:
            time_slots.append((create_date(times[0]),create_date(times[1])-create_date(times[0])))
        for comps in tracks[1]:
            comp_types.append(comps)  
        task_dates[st]=[time_slots, comp_types]
        customDates.append(time_slots)
    
    fig, ax = plt.subplots(figsize=(8,10))
    ax.patch.set_facecolor('lavender')
    
    # use a colormap
    cmap = plt.cm.Blues
    barHeight = len(ylabels)/3
    ticklist = []; 
    def drawLoadDuration(period, starty, compt, c1, c2, c3):
        if len(compt) >= 1:
            if compt[0] == 3:
                if c3 == 0:
                    ax.broken_barh((period), (starty, barHeight), facecolors='crimson', lw=0, zorder=2, alpha = 0.9, label='3-component'); c3 += 1
                else:
                    ax.broken_barh((period), (starty, barHeight), facecolors='crimson', lw=0, zorder=2, alpha = 0.9)
            
            if compt[0] == 1:
                if c1 == 0:
                    ax.broken_barh((period), (starty, barHeight), facecolors='mediumslateblue', lw=0, zorder=2, alpha = 0.9, label='1-component'); c1 += 1
                else:
                    ax.broken_barh((period), (starty, barHeight), facecolors='mediumslateblue', lw=0, zorder=2, alpha = 0.9)                          
                
            if compt[0] == 2 : 
                if c2 == 0:
                    ax.broken_barh((period), (starty, barHeight), facecolors='darkorange', lw=0, zorder=2, alpha = 0.9, label='2-component'); c2 += 1
                else:
                    ax.broken_barh((period), (starty, barHeight), facecolors='darkorange', lw=0, zorder=2, alpha = 0.9)
    
        ticklist.append(starty+barHeight/2.0)
        return c1, c2, c3
    
    
    h0 = 3; c1 = 0; c2=0; c3=0
    for st in ylabels:
        c1, c2, c3= drawLoadDuration(task_dates[st][0], h0, task_dates[st][1], c1, c2, c3)
        h0 += int(len(ylabels)*2)
        
    legend_properties = {'weight':'bold'}  
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0, h0)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Stations', fontsize=12)
    ax.set_yticks(ticklist)
    ax.tick_params('x', colors=cmap(1.))
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.grid(True)
    
    ax.xaxis_date() #Tell matplotlib that these are dates...
    rule = rrulewrapper(HOURLY, interval=time_interval)
    loc = RRuleLocator(rule)
    formatter = DateFormatter('%Y-%m-%d %H')
     
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(formatter)
    labelsx = ax.get_xticklabels()
    plt.setp(labelsx, rotation=30, fontsize=10)
     
    # Format the legend
    font = font_manager.FontProperties(size='small')
    ax.legend(loc=1, prop=font)
    plt.locator_params(axis='x', nbins=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.savefig('data_chart.png', dpi=300)
    plt.show()



def _date_convertor(r):  
          
    mls = r.split('.')
    if len(mls) == 1:
        new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
    else:
        new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
    return new_t