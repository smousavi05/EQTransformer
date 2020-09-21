#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:16:51 2019

@author: mostafamousavi
last update: 06/06/2020
"""
from __future__ import division, print_function
import numpy as np
import h5py
import matplotlib
matplotlib.use('agg')
from tqdm import tqdm
import keras
from keras import backend as K
from keras.layers import add, Activation, LSTM, Conv1D
from keras.layers import MaxPooling1D, UpSampling1D, Cropping1D, SpatialDropout1D, Bidirectional, BatchNormalization 
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from obspy.signal.trigger import trigger_onset
import matplotlib
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


class DataGenerator(keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Name of hdf5 file containing waveforms data.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32
        Batch size.
            
    n_channels: int, default=3
        Number of channels.
            
    phase_window: int, fixed=40
        The number of samples (window) around each phaset.
            
    shuffle: bool, default=True
        Shuffeling the list.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.
            
    label_type: str, default=gaussian 
        Labeling type: 'gaussian', 'triangle', or 'box'.
             
    augmentation: bool, default=True
        If True, half of each batch will be augmented version of the other half.
            
    add_event_r: {float, None}, default=None
        Chance for randomly adding a second event into the waveform.

    add_gap_r: {float, None}, default=None
        Add an interval with zeros into the waveform representing filled gaps.
            
    shift_event_r: {float, None}, default=0.9
        Rate of augmentation for randomly shifting the event within a trace. 
            
    add_noise_r: {float, None}, default=None
        Chance for randomly adding Gaussian noise into the waveform.
            
    drop_channe_r: {float, None}, default=None
        Chance for randomly dropping some of the channels.
            
    scale_amplitude_r: {float, None}, default=None
        Chance for randomly amplifying the waveform amplitude.

    pre_emphasis: bool, default=False
        If True, waveforms will be pre emphasized. 

    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
    
    """   
    
    def __init__(self, 
                 list_IDs, 
                 file_name, 
                 dim, 
                 batch_size=32, 
                 n_channels=3, 
                 phase_window= 40, 
                 shuffle=True, 
                 norm_mode = 'max',
                 label_type = 'gaussian',                 
                 augmentation = False, 
                 add_event_r = None,
                 add_gap_r = None,
                 shift_event_r = None,
                 add_noise_r = None, 
                 drop_channe_r = None, 
                 scale_amplitude_r = None, 
                 pre_emphasis = True):
       
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.phase_window = phase_window
        self.list_IDs = list_IDs
        self.file_name = file_name        
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.norm_mode = norm_mode
        self.label_type = label_type       
        self.augmentation = augmentation   
        self.add_event_r = add_event_r
        self.add_gap_r = add_gap_r  
        self.shift_event_r = shift_event_r
        self.add_noise_r = add_noise_r
        self.drop_channe_r = drop_channe_r
        self.scale_amplitude_r = scale_amplitude_r
        self.pre_emphasis = pre_emphasis


    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.augmentation:
            return 2*int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.augmentation:
            indexes = self.indexes[index*self.batch_size//2:(index+1)*self.batch_size//2]
            indexes = np.append(indexes, indexes)
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y1, y2, y3 = self.__data_generation(list_IDs_temp)
        return ({'input': X}, {'detector': y1, 'picker_P': y2, 'picker_S': y3})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  
    
    def _normalize(self, data, mode = 'max'):  
        'Normalize waveforms in each batch'
        
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data
    
    def _scale_amplitude(self, data, rate):
        'Scale amplitude or waveforms'
        
        tmp = np.random.uniform(0, 1)
        if tmp < rate:
            data *= np.random.uniform(1, 3)
        elif tmp < 2*rate:
            data /= np.random.uniform(1, 3)
        return data

    def _drop_channel(self, data, snr, rate):
        'Randomly replace values of one or two components to zeros in earthquake data'

        data = np.copy(data)
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _drop_channel_noise(self, data, rate):
        'Randomly replace values of one or two components to zeros in noise data'
        
        data = np.copy(data)
        if np.random.uniform(0, 1) < rate: 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _add_gaps(self, data, rate): 
        'Randomly add gaps (zeros) of different sizes into waveforms'
        
        data = np.copy(data)
        gap_start = np.random.randint(0, 4000)
        gap_end = np.random.randint(gap_start, 5500)
        if np.random.uniform(0, 1) < rate: 
            data[gap_start:gap_end,:] = 0           
        return data  
    
    def _add_noise(self, data, snr, rate):
        'Randomly add Gaussian noie with a random SNR into waveforms'
        
        data_noisy = np.empty((data.shape))
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
            data_noisy = np.empty((data.shape))
            data_noisy[:, 0] = data[:,0] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,0]), data.shape[0])
            data_noisy[:, 1] = data[:,1] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,1]), data.shape[0])
            data_noisy[:, 2] = data[:,2] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,2]), data.shape[0])    
        else:
            data_noisy = data
        return data_noisy   
         
    def _adjust_amplitude_for_multichannels(self, data):
        'Adjust the amplitude of multichaneel data'
        
        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert(tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
          data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def _label(self, a=0, b=20, c=40):  
        'Used for triangolar labeling'
        
        z = np.linspace(a, c, num = 2*(b-a)+1)
        y = np.zeros(z.shape)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half]-a) / (b-a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c-z[second_half]) / (c-b)
        return y

    def _add_event(self, data, addp, adds, coda_end, snr, rate): 
        'Add a scaled version of the event into the empty part of the trace'
       
        added = np.copy(data)
        additions = None
        spt_secondEV = None
        sst_secondEV = None
        if addp and adds:
            s_p = adds - addp
            if np.random.uniform(0, 1) < rate and all(snr>=10.0) and (data.shape[0]-s_p-21-coda_end) > 20:     
                secondEV_strt = np.random.randint(coda_end, data.shape[0]-s_p-21)
                scaleAM = 1/np.random.randint(1, 10)
                space = data.shape[0]-secondEV_strt  
                added[secondEV_strt:secondEV_strt+space, 0] += data[addp:addp+space, 0]*scaleAM
                added[secondEV_strt:secondEV_strt+space, 1] += data[addp:addp+space, 1]*scaleAM 
                added[secondEV_strt:secondEV_strt+space, 2] += data[addp:addp+space, 2]*scaleAM          
                spt_secondEV = secondEV_strt   
                if  spt_secondEV + s_p + 21 <= data.shape[0]:
                    sst_secondEV = spt_secondEV + s_p
                if spt_secondEV and sst_secondEV:                                                                     
                    additions = [spt_secondEV, sst_secondEV] 
                    data = added
                 
        return data, additions    
    
    
    def _shift_event(self, data, addp, adds, coda_end, snr, rate): 
        'Randomly rotate the array to shift the event location'
        
        org_len = len(data)
        data2 = np.copy(data)
        addp2 = adds2 = coda_end2 = None;
        if np.random.uniform(0, 1) < rate:             
            nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
            data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
            data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
            data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]
                    
            if addp+nrotate >= 0 and addp+nrotate < org_len:
                addp2 = addp+nrotate;
            else:
                addp2 = None;
            if adds+nrotate >= 0 and adds+nrotate < org_len:               
                adds2 = adds+nrotate;
            else:
                adds2 = None;                   
            if coda_end+nrotate < org_len:                              
                coda_end2 = coda_end+nrotate 
            else:
                coda_end2 = org_len                 
            if addp2 and adds2:
                data = data2;
                addp = addp2;
                adds = adds2;
                coda_end= coda_end2;                                      
        return data, addp, adds, coda_end      
    
    def _pre_emphasis(self, data, pre_emphasis=0.97):
        'apply the pre_emphasis'

        for ch in range(self.n_channels): 
            bpf = data[:, ch]  
            data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data
                    
    def __data_generation(self, list_IDs_temp):
        'read the waveforms'         
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        y1 = np.zeros((self.batch_size, self.dim, 1))
        y2 = np.zeros((self.batch_size, self.dim, 1))
        y3 = np.zeros((self.batch_size, self.dim, 1))
        fl = h5py.File(self.file_name, 'r')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            additions = None
            dataset = fl.get('data/'+str(ID))

            if ID.split('_')[-1] == 'EV':
                data = np.array(dataset)                    
                spt = int(dataset.attrs['p_arrival_sample']);
                sst = int(dataset.attrs['s_arrival_sample']);
                coda_end = int(dataset.attrs['coda_end_sample']);
                snr = dataset.attrs['snr_db'];
                    
            elif ID.split('_')[-1] == 'NO':
                data = np.array(dataset)
           
            ## augmentation 
            if self.augmentation == True:                 
                if i <= self.batch_size//2:   
                    if self.shift_event_r and dataset.attrs['trace_category'] == 'earthquake_local':
                        data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2);                                       
                    if self.norm_mode:                    
                        data = self._normalize(data, self.norm_mode)  
                else:                  
                    if dataset.attrs['trace_category'] == 'earthquake_local':                   
                        if self.shift_event_r:
                            data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r); 
                            
                        if self.add_event_r:
                            data, additions = self._add_event(data, spt, sst, coda_end, snr, self.add_event_r); 
                                
                        if self.add_noise_r:
                            data = self._add_noise(data, snr, self.add_noise_r);
    
                        if self.drop_channe_r:    
                            data = self._drop_channel(data, snr, self.drop_channe_r);
                            data = self._adjust_amplitude_for_multichannels(data)  
                                    
                        if self.scale_amplitude_r:
                            data = self._scale_amplitude(data, self.scale_amplitude_r); 
                                    
                        if self.pre_emphasis:  
                            data = self._pre_emphasis(data) 
                                    
                        if self.norm_mode:    
                            data = self._normalize(data, self.norm_mode)                            
                                    
                    elif dataset.attrs['trace_category'] == 'noise':
                        if self.drop_channe_r:    
                            data = self._drop_channel_noise(data, self.drop_channe_r);
                            
                        if self.add_gap_r:    
                            data = self._add_gaps(data, self.add_gap_r)
                            
                        if self.norm_mode: 
                            data = self._normalize(data, self.norm_mode) 

            elif self.augmentation == False:  
                if self.shift_event_r and dataset.attrs['trace_category'] == 'earthquake_local':
                    data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2);                     
                if self.norm_mode:                    
                    data = self._normalize(data, self.norm_mode)                          

            X[i, :, :] = data                                       

            ## labeling 
            if dataset.attrs['trace_category'] == 'earthquake_local': 
                if self.label_type  == 'gaussian': 
                    sd = None    
                    if spt and sst: 
                        sd = sst - spt  
                                                            
                    if sd and sst:
                        if sst+int(0.4*sd) <= self.dim: 
                            y1[i, spt:int(sst+(0.4*sd)), 0] = 1        
                        else:
                            y1[i, spt:self.dim, 0] = 1                       
                             
                    if spt and (spt-20 >= 0) and (spt+20 < self.dim):
                        y2[i, spt-20:spt+20, 0] = np.exp(-(np.arange(spt-20,spt+20)-spt)**2/(2*(10)**2))[:self.dim-(spt-20)]                
                    elif spt and (spt-20 < self.dim):
                        y2[i, 0:spt+20, 0] = np.exp(-(np.arange(0,spt+20)-spt)**2/(2*(10)**2))[:self.dim-(spt-20)]
    
                    if sst and (sst-20 >= 0) and (sst-20 < self.dim):
                        y3[i, sst-20:sst+20, 0] = np.exp(-(np.arange(sst-20,sst+20)-sst)**2/(2*(10)**2))[:self.dim-(sst-20)]
                    elif sst and (sst-20 < self.dim):
                        y3[i, 0:sst+20, 0] = np.exp(-(np.arange(0,sst+20)-sst)**2/(2*(10)**2))[:self.dim-(sst-20)]

                    if additions: 
                        add_sd = None
                        add_spt = additions[0];
                        add_sst = additions[1];
                        if add_spt and add_sst: 
                            add_sd = add_sst - add_spt  
                                    
                        if add_sd and add_sst+int(0.4*add_sd) <= self.dim: 
                            y1[i, add_spt:int(add_sst+(0.4*add_sd)), 0] = 1        
                        else:
                            y1[i, add_spt:self.dim, 0] = 1
                            
                        if add_spt and (add_spt-20 >= 0) and (add_spt+20 < self.dim):
                            y2[i, add_spt-20:add_spt+20, 0] = np.exp(-(np.arange(add_spt-20,add_spt+20)-add_spt)**2/(2*(10)**2))[:self.dim-(add_spt-20)]
                        elif add_spt and (add_spt+20 < self.dim):
                            y2[i, 0:add_spt+20, 0] = np.exp(-(np.arange(0,add_spt+20)-add_spt)**2/(2*(10)**2))[:self.dim-(add_spt-20)]
        
                        if add_sst and (add_sst-20 >= 0) and (add_sst+20 < self.dim):
                            y3[i, add_sst-20:add_sst+20, 0] = np.exp(-(np.arange(add_sst-20,add_sst+20)-add_sst)**2/(2*(10)**2))[:self.dim-(add_sst-20)]
                        elif add_sst and (add_sst+20 < self.dim):
                            y3[i, 0:add_sst+20, 0] = np.exp(-(np.arange(0,add_sst+20)-add_sst)**2/(2*(10)**2))[:self.dim-(add_sst-20)]
                                                    

                elif self.label_type  == 'triangle':                      
                    sd = None    
                    if spt and sst: 
                        sd = sst - spt  
                                                            
                    if sd and sst:
                        if sst+int(0.4*sd) <= self.dim: 
                            y1[i, spt:int(sst+(0.4*sd)), 0] = 1        
                        else:
                            y1[i, spt:self.dim, 0] = 1                     
                        
                    if spt and (spt-20 >= 0) and (spt+21 < self.dim):
                        y2[i, spt-20:spt+21, 0] = self._label()
                    elif spt and (spt+21 < self.dim):
                        y2[i, 0:spt+spt+1, 0] = self._label(a=0, b=spt, c=2*spt)
                    elif spt and (spt-20 >= 0):
                        pdif = self.dim - spt
                        y2[i, spt-pdif-1:self.dim, 0] = self._label(a=spt-pdif, b=spt, c=2*pdif)
         
                    if sst and (sst-20 >= 0) and (sst+21 < self.dim):
                        y3[i, sst-20:sst+21, 0] = self._label()
                    elif sst and (sst+21 < self.dim):
                        y3[i, 0:sst+sst+1, 0] = self._label(a=0, b=sst, c=2*sst)
                    elif sst and (sst-20 >= 0):
                        sdif = self.dim - sst
                        y3[i, sst-sdif-1:self.dim, 0] = self._label(a=sst-sdif, b=sst, c=2*sdif)             
    
                    if additions: 
                        add_spt = additions[0];
                        add_sst = additions[1];
                        add_sd = None
                        if add_spt and add_sst: 
                            add_sd = add_sst - add_spt                     
                        
                        if add_sd and add_sst+int(0.4*add_sd) <= self.dim: 
                            y1[i, add_spt:int(add_sst+(0.4*add_sd)), 0] = 1        
                        else:
                            y1[i, add_spt:self.dim, 0] = 1                     
    
                        if add_spt and (add_spt-20 >= 0) and (add_spt+21 < self.dim):
                            y2[i, add_spt-20:add_spt+21, 0] = self._label()
                        elif add_spt and (add_spt+21 < self.dim):
                            y2[i, 0:add_spt+add_spt+1, 0] = self._label(a=0, b=add_spt, c=2*add_spt)
                        elif add_spt and (add_spt-20 >= 0):
                            pdif = self.dim - add_spt
                            y2[i, add_spt-pdif-1:self.dim, 0] = self._label(a=add_spt-pdif, b=add_spt, c=2*pdif)
    
                        if add_sst and (add_sst-20 >= 0) and (add_sst+21 < self.dim):
                            y3[i, add_sst-20:add_sst+21, 0] = self._label()
                        elif add_sst and (add_sst+21 < self.dim):
                            y3[i, 0:add_sst+add_sst+1, 0] = self._label(a=0, b=add_sst, c=2*add_sst)
                        elif add_sst and (add_sst-20 >= 0):
                            sdif = self.dim - add_sst
                            y3[i, add_sst-sdif-1:self.dim, 0] = self._label(a=add_sst-sdif, b=add_sst, c=2*sdif) 
    
    
                elif self.label_type  == 'box':
                    sd = None                             
                    if sst and spt:
                        sd = sst - spt      

                    if sd and sst+int(0.4*sd) <= self.dim: 
                        y1[i, spt:int(sst+(0.4*sd)), 0] = 1        
                    else:
                        y1[i, spt:self.dim, 0] = 1         
                    if spt: 
                        y2[i, spt-20:spt+20, 0] = 1
                    if sst:
                        y3[i, sst-20:sst+20, 0] = 1                       
                   
                    if additions:
                        add_sd = None
                        add_spt = additions[0];
                        add_sst = additions[1];
                        if add_spt and add_sst:
                            add_sd = add_sst - add_spt  
                            
                        if add_sd and add_sst+int(0.4*add_sd) <= self.dim: 
                            y1[i, add_spt:int(add_sst+(0.4*add_sd)), 0] = 1        
                        else:
                            y1[i, add_spt:self.dim, 0] = 1                     
                        if add_spt:
                            y2[i, add_spt-20:add_spt+20, 0] = 1
                        if add_sst:
                            y3[i, add_sst-20:add_sst+20, 0] = 1                 

        fl.close() 
                           
        return X, y1.astype('float32'), y2.astype('float32'), y3.astype('float32')





class PreLoadGenerator(keras.utils.Sequence):
    
    """ 
    Keras generator with preprocessing. Pre-load version. 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    inp_data: dic
        A dictionary of input hdf5 datasets.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32
        Batch size.
            
    n_channels: int, default=3
        Number of channels.
            
    phase_window: int, fixed=40
        The number of samples (window) around each phaset.
            
    shuffle: bool, default=True
        Shuffeling the list.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.
            
    label_type: str, default=gaussian 
        Labeling type. 'gaussian', 'triangle', or 'box'.
             
    augmentation: bool, default=True
        If True, half of each batch will be augmented version of the other half.
            
    add_event_r: {float, None}, default=None
        Chance for randomly adding a second event into the waveform.

    add_gap_r: {float, None}, default=None
        Add an interval with zeros into the waveform representing filled gaps.
            
    shift_event_r: {float, None}, default=0.9
        Rate of augmentation for randomly shifting the event within a trace. 
            
    add_noise_r: {float, None}, default=None
        Chance for randomly adding Gaussian noise into the waveform.
            
    drop_channe_r: {float, None}, default=None
        Chance for randomly dropping some of the channels.
            
    scale_amplitude_r: {float, None}, default=None
        Chance for randomly amplifying the waveform amplitude.
            
    pre_emphasis: bool, default=False
        If True, waveforms will be pre emphasized. 

    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
   
    """   

    def __init__(self, 
                 inp_data,
                 list_IDs, 
                 file_name, 
                 dim, 
                 batch_size=32, 
                 n_channels=3, 
                 phase_window= 40, 
                 shuffle=True, 
                 norm_mode = 'max',
                 label_type = 'gaussian',                 
                 augmentation = False, 
                 add_event_r = None,
                 add_gap_r = None,
                 shift_event_r = None,
                 add_noise_r = None, 
                 drop_channe_r = None, 
                 scale_amplitude_r = None, 
                 pre_emphasis = True):
       
        'Initialization'
        self.inp_data =inp_data
        self.dim = dim
        self.batch_size = batch_size
        self.phase_window = phase_window
        self.list_IDs = list_IDs
        self.file_name = file_name        
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.norm_mode = norm_mode
        self.label_type = label_type       
        self.augmentation = augmentation   
        self.add_event_r = add_event_r
        self.add_gap_r = add_gap_r  
        self.shift_event_r = shift_event_r
        self.add_noise_r = add_noise_r
        self.drop_channe_r = drop_channe_r
        self.scale_amplitude_r = scale_amplitude_r
        self.pre_emphasis = pre_emphasis       
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        
        if self.augmentation:
            return 2*int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        
        if self.augmentation:
            indexes = self.indexes[index*self.batch_size//2:(index+1)*self.batch_size//2]
            indexes = np.append(indexes, indexes)
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y1, y2, y3 = self.__data_generation(list_IDs_temp)
        return ({'input': X}, {'detector': y1, 'picker_P': y2, 'picker_S': y3})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  
    
    def _normalize(self, data, mode = 'max'):   
        'Normalize waveforms in each batch'

        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data
    
    def _scale_amplitude(self, data, rate):
        'Scale amplitude or waveforms'
        
        tmp = np.random.uniform(0, 1)
        if tmp < rate:
            data *= np.random.uniform(1, 3)
        elif tmp < 2*rate:
            data /= np.random.uniform(1, 3)
        return data

    def _drop_channel(self, data, snr, rate):
        'Randomly replace values of one or two components to zeros in earthquake data'
        
        data = np.copy(data)
        if np.random.uniform(0, 1) < rate and all(snr >= 10): 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _drop_channel_noise(self, data, rate):
        'Randomly replace values of one or two components to zeros in noise data'
        
        data = np.copy(data)
        if np.random.uniform(0, 1) < rate: 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _add_gaps(self, data, rate): 
        'Randomly add gaps (zeros) of different sizes into waveforms'
        
        data = np.copy(data)
        gap_start = np.random.randint(0, 4000)
        gap_end = np.random.randint(gap_start, 5500)
        if np.random.uniform(0, 1) < rate: 
            data[gap_start:gap_end,:] = 0           
        return data  
    
    def _add_noise(self, data, snr, rate):
        'Randomly add Gaussian noie with a random SNR into waveforms'
        
        data_noisy = np.empty((data.shape))
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
            data_noisy = np.empty((data.shape))
            data_noisy[:, 0] = data[:,0] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,0]), data.shape[0])
            data_noisy[:, 1] = data[:,1] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,1]), data.shape[0])
            data_noisy[:, 2] = data[:,2] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,2]), data.shape[0])    
        else:
            data_noisy = data
        return data_noisy   
         
    def _adjust_amplitude_for_multichannels(self, data):
        'Adjust the amplitude of multichaneel data'
        
        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert(tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
          data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def _label(self, a=0, b=20, c=40):  
        'Used for triangolar labeling'
        
        z = np.linspace(a, c, num = 2*(b-a)+1)
        y = np.zeros(z.shape)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half]-a) / (b-a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c-z[second_half]) / (c-b)
        return y

    def _add_event(self, data, addp, adds, coda_end, snr, rate): 
        'Add a scaled version of the event into the empty part of the trace'
        
        added = np.copy(data)
        additions = None
        spt_secondEV = None
        sst_secondEV = None
        if addp and adds:
            s_p = adds - addp
            if np.random.uniform(0, 1) < rate and all(snr >= 10.0) and (data.shape[0]-s_p-21-coda_end) > 20:     
                secondEV_strt = np.random.randint(coda_end, data.shape[0]-s_p-21)
                scaleAM = 1/np.random.randint(1, 10)
                space = data.shape[0]-secondEV_strt  
                added[secondEV_strt:secondEV_strt+space, 0] += data[addp:addp+space, 0]*scaleAM
                added[secondEV_strt:secondEV_strt+space, 1] += data[addp:addp+space, 1]*scaleAM 
                added[secondEV_strt:secondEV_strt+space, 2] += data[addp:addp+space, 2]*scaleAM          
                spt_secondEV = secondEV_strt   
                if  spt_secondEV + s_p + 21 <= data.shape[0]:
                    sst_secondEV = spt_secondEV + s_p
                if spt_secondEV and sst_secondEV:                                                                     
                    additions = [spt_secondEV, sst_secondEV] 
                    data = added
                 
        return data, additions    
    
    
    def _shift_event(self, data, addp, adds, coda_end, snr, rate): 
        'Randomly rotate the array to shift the event location'
        
        org_len = len(data)
        data2 = np.copy(data)
        addp2 = adds2 = coda_end2 = None;
        if np.random.uniform(0, 1) < rate:             
            nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
            data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
            data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
            data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]
                    
            if addp+nrotate >= 0 and addp+nrotate < org_len:
                addp2 = addp+nrotate;
            else:
                addp2 = None;
            if adds+nrotate >= 0 and adds+nrotate < org_len:               
                adds2 = adds+nrotate;
            else:
                adds2 = None;                   
            if coda_end+nrotate < org_len:                              
                coda_end2 = coda_end+nrotate 
            else:
                coda_end2 = org_len                 
            if addp2 and adds2:
                data = data2;
                addp = addp2;
                adds = adds2;
                coda_end= coda_end2;                                      
        return data, addp, adds, coda_end      
  
    
    
    def _pre_emphasis(self, data, pre_emphasis=0.97):
        'apply the pre_emphasis'
        
        for ch in range(self.n_channels): 
            bpf = data[:, ch]  
            data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data
                    
    def __data_generation(self, list_IDs_temp):
        'readint the waveforms' 
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        y1 = np.zeros((self.batch_size, self.dim, 1))
        y2 = np.zeros((self.batch_size, self.dim, 1))
        y3 = np.zeros((self.batch_size, self.dim, 1))            
        # Generate data
        for i, ID in enumerate(list_IDs_temp):            
            additions = None
            dataset = self.inp_data[ID]
            data = np.array(dataset) 
            if dataset.attrs['trace_category'] == 'earthquake_local':                   
                spt = int(dataset.attrs['p_arrival_sample']);
                sst = int(dataset.attrs['s_arrival_sample']);
                coda_end = int(dataset.attrs['coda_end_sample']);
                snr = dataset.attrs['snr_db'];
                
            if self.augmentation == True:                 
                if i <= self.batch_size//2:                                         
                    if self.shift_event_r and dataset.attrs['trace_category'] == 'earthquake_local':
                        data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2); 
                    if self.norm_mode:                    
                        data = self._normalize(data, self.norm_mode)                            
                else:                  
                    if dataset.attrs['trace_category'] == 'earthquake_local':  
                        if self.shift_event_r and spt:
                            data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r); 
                                             
                        if self.add_event_r and spt and sst:
                            data, additions = self._add_event(data, spt, sst, coda_end, snr, self.add_event_r); 
                            
                        if self.add_noise_r:
                            data = self._add_noise(data, snr, self.add_noise_r);

                        if self.drop_channe_r:    
                            data = self._drop_channel(data, snr, self.drop_channe_r);
                            data = self._adjust_amplitude_for_multichannels(data)  
                                
                        if self.scale_amplitude_r:
                            data = self._scale_amplitude(data, self.scale_amplitude_r); 
                                
                        if self.pre_emphasis:  
                            data = self._pre_emphasis(data) 
                                
                        if self.norm_mode:    
                            data = self._normalize(data, self.norm_mode)

                    elif dataset.attrs['trace_category'] == 'noise':
                        if self.drop_channe_r:    
                            data = self._drop_channel_noise(data, self.drop_channe_r);
                            
                        if self.add_gap_r:    
                            data = self._add_gaps(data, self.add_gap_r)
                            
                        if self.norm_mode: 
                            data = self._normalize(data, self.norm_mode) 
                            
            elif self.augmentation == False:  
                if self.shift_event_r and dataset.attrs['trace_category'] == 'earthquake_local':
                    data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2);                 
                if self.norm_mode:                    
                    data = self._normalize(data, self.norm_mode)    
                      
            X[i, :, :] = data                                                           
            ## labeling             
            if dataset.attrs['trace_category'] == 'earthquake_local':
 
                if self.label_type  == 'gaussian': 
                    sd = None                             
                    if spt and sst: 
                        sd = sst - spt  
                                                            
                    if sd and sst:
                        if sst+int(1.4*sd) <= self.dim: 
                            y1[i, spt:int(sst+(1.4*sd)), 0] = 1        
                        else:
                            y1[i, spt:self.dim, 0] = 1                     
                                 
                    if spt and (spt-20 >= 0) and (spt+20 < self.dim):
                        y2[i, spt-20:spt+20, 0] = np.exp(-(np.arange(spt-20,spt+20)-spt)**2/(2*(10)**2))[:self.dim-(spt-20)]                
                    elif spt and (spt-20 < self.dim):
                        y2[i, 0:spt+20, 0] = np.exp(-(np.arange(0,spt+20)-spt)**2/(2*(10)**2))[:self.dim-(spt-20)]
    
                    if sst and (sst-20 >= 0) and (sst-20 < self.dim):
                        y3[i, sst-20:sst+20, 0] = np.exp(-(np.arange(sst-20,sst+20)-sst)**2/(2*(10)**2))[:self.dim-(sst-20)]
                    elif sst and (sst-20 < self.dim):
                        y3[i, 0:sst+20, 0] = np.exp(-(np.arange(0,sst+20)-sst)**2/(2*(10)**2))[:self.dim-(sst-20)]

                    if additions: 
                        add_spt = additions[0];
                        add_sst = additions[1];
                        add_sd = None
                        
                        if add_spt and add_sst: 
                            add_sd = add_sst - add_spt  
                                    
                        if add_sd and add_sst+int(1.4*add_sd) <= self.dim: 
                            y1[i, add_spt:int(add_sst+(1.4*add_sd)), 0] = 1        
                        else:
                            y1[i, add_spt:self.dim, 0] = 1
                            
                        if add_spt and (add_spt-20 >= 0) and (add_spt+20 < self.dim):
                            y2[i, add_spt-20:add_spt+20, 0] = np.exp(-(np.arange(add_spt-20,add_spt+20)-add_spt)**2/(2*(10)**2))[:self.dim-(add_spt-20)]
                        elif add_spt and (add_spt+20 < self.dim):
                            y2[i, 0:add_spt+20, 0] = np.exp(-(np.arange(0,add_spt+20)-add_spt)**2/(2*(10)**2))[:self.dim-(add_spt-20)]
        
                        if add_sst and (add_sst-20 >= 0) and (add_sst+20 < self.dim):
                            y3[i, add_sst-20:add_sst+20, 0] = np.exp(-(np.arange(add_sst-20,add_sst+20)-add_sst)**2/(2*(10)**2))[:self.dim-(add_sst-20)]
                        elif add_sst and (add_sst+20 < self.dim):
                            y3[i, 0:add_sst+20, 0] = np.exp(-(np.arange(0,add_sst+20)-add_sst)**2/(2*(10)**2))[:self.dim-(add_sst-20)]
                                                    
                elif self.label_type  == 'triangle':  
                    sd = None     
                    if spt and sst: 
                        sd = sst - spt  
                                                            
                    if sd and sst:
                        if sst+int(1.4*sd) <= self.dim: 
                            y1[i, spt:int(sst+(1.4*sd)), 0] = 1        
                        else:
                            y1[i, spt:self.dim, 0] = 1                     
                        
                    if spt and (spt-20 >= 0) and (spt+21 < self.dim):
                        y2[i, spt-20:spt+21, 0] = self._label()
                    elif spt and (spt+21 < self.dim):
                        y2[i, 0:spt+spt+1, 0] = self._label(a=0, b=spt, c=2*spt)
                    elif spt and (spt-20 >= 0):
                        pdif = self.dim - spt
                        y2[i, spt-pdif-1:self.dim, 0] = self._label(a=spt-pdif, b=spt, c=2*pdif)
         
                    if sst and (sst-20 >= 0) and (sst+21 < self.dim):
                        y3[i, sst-20:sst+21, 0] = self._label()
                    elif sst and (sst+21 < self.dim):
                        y3[i, 0:sst+sst+1, 0] = self._label(a=0, b=sst, c=2*sst)
                    elif sst and (sst-20 >= 0):
                        sdif = self.dim - sst
                        y3[i, sst-sdif-1:self.dim, 0] = self._label(a=sst-sdif, b=sst, c=2*sdif)    
         
                    if additions: 
                        add_spt = additions[0];
                        add_sst = additions[1];
                        add_sd = None
                        
                        if add_spt and add_sst: 
                            add_sd = add_sst - add_spt                     
                        
                        if add_sd and add_sst+int(1.4*add_sd) <= self.dim: 
                            y1[i, add_spt:int(add_sst+(1.4*add_sd)), 0] = 1        
                        else:
                            y1[i, add_spt:self.dim, 0] = 1                     
    
                        if add_spt and (add_spt-20 >= 0) and (add_spt+21 < self.dim):
                            y2[i, add_spt-20:add_spt+21, 0] = self.label()
                        elif add_spt and (add_spt+21 < self.dim):
                            y2[i, 0:add_spt+add_spt+1, 0] = self.label(a=0, b=add_spt, c=2*add_spt)
                        elif add_spt and (add_spt-20 >= 0):
                            pdif = self.dim - add_spt
                            y2[i, add_spt-pdif-1:self.dim, 0] = self.label(a=add_spt-pdif, b=add_spt, c=2*pdif)
    
                        if add_sst and (add_sst-20 >= 0) and (add_sst+21 < self.dim):
                            y3[i, add_sst-20:add_sst+21, 0] = self.label()
                        elif add_sst and (add_sst+21 < self.dim):
                            y3[i, 0:add_sst+add_sst+1, 0] = self.label(a=0, b=add_sst, c=2*add_sst)
                        elif add_sst and (add_sst-20 >= 0):
                            sdif = self.dim - add_sst
                            y3[i, add_sst-sdif-1:self.dim, 0] = self.label(a=add_sst-sdif, b=add_sst, c=2*sdif) 
    
                elif self.label_type  == 'box':
                    sd = None
                    if sst and spt:
                        sd = sst - spt      

                    if sd and sst+int(1.4*sd) <= self.dim: 
                        y1[i, spt:int(sst+(1.4*sd)), 0] = 1        
                    else:
                        y1[i, spt:self.dim, 0] = 1
                        
                    if spt: 
                        y2[i, spt-20:spt+20, 0] = 1
                    if sst:
                        y3[i, sst-20:sst+20, 0] = 1                       
                   
                    if additions: 
                        add_sd = None
                        add_spt = additions[0];
                        add_sst = additions[1];
                        if add_spt and add_sst:
                            add_sd = add_sst - add_spt  
                            
                        if add_sd and add_sst+int(1.4*add_sd) <= self.dim: 
                            y1[i, add_spt:int(add_sst+(1.4*add_sd)), 0] = 1        
                        else:
                            y1[i, add_spt:self.dim, 0] = 1                     
                        if add_spt:
                            y2[i, add_spt-20:add_spt+20, 0] = 1
                        if add_sst:
                            y3[i, add_sst-20:add_sst+20, 0] = 1                 
                                                              
        return X.astype('float32'), y1.astype('float32'), y2.astype('float32'), y3.astype('float32')



def data_reader( list_IDs, 
                 file_name, 
                 dim=6000, 
                 n_channels=3, 
                 norm_mode='max',
                 augmentation=False, 
                 add_event_r=None,
                 add_gap_r=None, 
                 shift_event_r=None,                                  
                 add_noise_r=None, 
                 drop_channe_r=None, 
                 scale_amplitude_r=None, 
                 pre_emphasis=True):   
    
    """ 
    
    For pre-processing and loading of data into memory. 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 datasets.
            
    dim: int, default=6000
        Dimension of input traces, in sample. 
           
    n_channels: int, default=3
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.
            
    augmentation: bool, default=True
        If True, half of each batch will be augmented version of the other half.
            
    add_event_r: {float, None}, default=None
        Chance for randomly adding a second event into the waveform.
            
    shift_event_r: {float, None}, default=0.9
        Rate of augmentation for randomly shifting the event within a trace. 
            
    add_noise_r: {float, None}, default=None
        Chance for randomly adding Gaussian noise into the waveform.
            
    drop_channe_r: {float, None}, default=None
        Chance for randomly dropping some of the channels.
            
    scale_amplitude_r: {float, None}, default=None
        Chance for randomly amplifying the waveform amplitude.
            
    pre_emphasis: bool, default=False
        If True, waveforms will be pre emphasized. 

    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
            
    Note
    -----
    Label type is fixed to box.
    
        
    """  
    
    def _normalize( data, mode = 'max'):
        'Normalize waveforms in each batch'
          
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data
    
    def _scale_amplitude( data, rate):
        'Scale amplitude or waveforms'
        
        tmp = np.random.uniform(0, 1)
        if tmp < rate:
            data *= np.random.uniform(1, 3)
        elif tmp < 2*rate:
            data /= np.random.uniform(1, 3)
        return data

    def _drop_channel( data, snr, rate):
        'Randomly replace values of one or two components to zeros in earthquake data'
        
        data = np.copy(data)
        if np.random.uniform(0, 1) < rate and all(snr >= 10): 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _drop_channel_noise(data, rate):
        'Randomly replace values of one or two components to zeros in noise data'
        
        data = np.copy(data)
        if np.random.uniform(0, 1) < rate: 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _add_gaps(data, rate): 
        'Randomly add gaps (zeros) of different sizes into waveforms'
        
        data = np.copy(data)
        gap_start = np.random.randint(0, 4000)
        gap_end = np.random.randint(gap_start, 5500)
        if np.random.uniform(0, 1) < rate: 
            data[gap_start:gap_end,:] = 0           
        return data  
    
    def _add_noise(data, snr, rate):
        'Randomly add Gaussian noie with a random SNR into waveforms'
        
        data_noisy = np.empty((data.shape))
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
            data_noisy = np.empty((data.shape))
            data_noisy[:, 0] = data[:,0] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,0]), data.shape[0])
            data_noisy[:, 1] = data[:,1] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,1]), data.shape[0])
            data_noisy[:, 2] = data[:,2] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,2]), data.shape[0])   
        else:
            data_noisy = data
        return data_noisy    
         
    def _adjust_amplitude_for_multichannels(data):
        'Adjust the amplitude of multichaneel data'
        
        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert(tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
          data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def _label(a=0, b=20, c=40): 
        'Used for triangolar labeling'
        
        z = np.linspace(a, c, num = 2*(b-a)+1)
        y = np.zeros(z.shape)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half]-a) / (b-a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c-z[second_half]) / (c-b)
        return y
    
    def _add_event(data, addp, adds, coda_end, snr, rate): 
        'Add a scaled version of the event into the empty part of the trace'
        
        added = np.copy(data)
        additions = spt_secondEV = sst_secondEV = None
        if addp and adds:
            s_p = adds - addp
            if np.random.uniform(0, 1) < rate and all(snr >= 10.0) and (data.shape[0]-s_p-21-coda_end) > 20: 
                secondEV_strt = np.random.randint(coda_end, data.shape[0]-s_p-21)
                scaleAM = 1/np.random.randint(1, 10)
                space = data.shape[0]-secondEV_strt  
                added[secondEV_strt:secondEV_strt+space, 0] += data[addp:addp+space, 0]*scaleAM
                added[secondEV_strt:secondEV_strt+space, 1] += data[addp:addp+space, 1]*scaleAM 
                added[secondEV_strt:secondEV_strt+space, 2] += data[addp:addp+space, 2]*scaleAM          
                spt_secondEV = secondEV_strt   
                if  spt_secondEV + s_p + 21 <= data.shape[0]:
                    sst_secondEV = spt_secondEV + s_p
                if spt_secondEV and sst_secondEV:                                                                     
                    additions = [spt_secondEV, sst_secondEV] 
                    data = added                
        return data, additions 



    def _shift_event(data, addp, adds, coda_end, snr, rate): 
        'Randomly rotate the array to shift the event location'
        
        org_len = len(data)
        data2 = np.copy(data)
        addp2 = adds2 = coda_end2 = None;
        if np.random.uniform(0, 1) < rate:             
            nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
            data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
            data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
            data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]
                    
            if addp+nrotate >= 0 and addp+nrotate < org_len:
                addp2 = addp+nrotate;
            else:
                addp2 = None;
            if adds+nrotate >= 0 and adds+nrotate < org_len:               
                adds2 = adds+nrotate;
            else:
                adds2 = None;                   
            if coda_end+nrotate < org_len:                              
                coda_end2 = coda_end+nrotate 
            else:
                coda_end2 = org_len                 
            if addp2 and adds2:
                data = data2;
                addp = addp2;
                adds = adds2;
                coda_end= coda_end2;                                      
        return data, addp, adds, coda_end   
    
    def _pre_emphasis( data, pre_emphasis=0.97):
        'apply the pre_emphasis'
        
        for ch in range(n_channels): 
            bpf = data[:, ch]  
            data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data
                    
    fl = h5py.File(file_name, 'r')

    if augmentation:
        X = np.zeros((2*len(list_IDs), dim, n_channels))
        y1 = np.zeros((2*len(list_IDs), dim, 1))
        y2 = np.zeros((2*len(list_IDs), dim, 1))
        y3 = np.zeros((2*len(list_IDs), dim, 1))
    else:
        X = np.zeros((len(list_IDs), dim, n_channels))
        y1 = np.zeros((len(list_IDs), dim, 1))
        y2 = np.zeros((len(list_IDs), dim, 1))
        y3 = np.zeros((len(list_IDs), dim, 1))     

    # Generate data
    pbar = tqdm(total=len(list_IDs)) 
    for i, ID in enumerate(list_IDs):
        pbar.update()

        additions = None
        dataset = fl.get('data/'+str(ID))
        
        if ID.split('_')[-1] == 'EV':            
            data = np.array(dataset)                    
            spt = int(dataset.attrs['p_arrival_sample']);
            sst = int(dataset.attrs['s_arrival_sample']);
            coda_end = int(dataset.attrs['coda_end_sample']);
            snr = dataset.attrs['snr_db'];
                    
        elif ID.split('_')[-1] == 'NO':
            data = np.array(dataset)
           
        if augmentation:                 
            if dataset.attrs['trace_category'] == 'earthquake_local':                   
                data, spt, sst, coda_end = _shift_event(data, spt, sst, coda_end, snr, shift_event_r/2); 
            if norm_mode: 
                data1 = _normalize(data, norm_mode)   
                          
            if dataset.attrs['trace_category'] == 'earthquake_local':
                if shift_event_r and spt:
                    data, spt, sst, coda_end = _shift_event(data, spt, sst, coda_end, snr, shift_event_r);  
                          
                if add_event_r:
                    data, additions = _add_event(data, spt, sst, coda_end, snr, add_event_r); 
                    
                if drop_channe_r:    
                    data = _drop_channel(data, snr, drop_channe_r);
                  #  data = _adjust_amplitude_for_multichannels(data); 
                          
                if scale_amplitude_r:
                    data = _scale_amplitude(data, scale_amplitude_r); 
                    
                if pre_emphasis:  
                    data = _pre_emphasis(data);

                if add_noise_r:
                    data = _add_noise(data, snr, add_noise_r);
                    
                if norm_mode:    
                    data2 = _normalize(data, norm_mode); 
                     
                            
            if dataset.attrs['trace_category'] == 'noise':
                if drop_channe_r:    
                    data = _drop_channel_noise(data, drop_channe_r);
                if add_gap_r:    
                    data = _add_gaps(data, add_gap_r)                    
                if norm_mode:                    
                    data2 = _normalize(data, norm_mode) 
                    
            X[i, :, :] = data1 
            X[len(list_IDs)+i, :, :] = data2                                      

            if dataset.attrs['trace_category'] == 'earthquake_local': 

                if spt and (spt-20 >= 0) and (spt+21 < dim):
                    y2[i, spt-20:spt+21, 0] = _label()
                    y2[len(list_IDs)+i, spt-20:spt+21, 0] = _label()                   
                elif spt and (spt+21 < dim):
                    y2[i, 0:spt+spt+1, 0] = _label(a=0, b=spt, c=2*spt)
                    y2[len(list_IDs)+i, 0:spt+spt+1, 0] = _label(a=0, b=spt, c=2*spt)                   
                elif spt and (spt-20 >= 0):
                    pdif = dim - spt
                    y2[i, spt-pdif-1:dim, 0] = _label(a=spt-pdif, b=spt, c=2*pdif)
                    y2[len(list_IDs)+i, spt-pdif-1:dim, 0] = _label(a=spt-pdif, b=spt, c=2*pdif)
         
                if sst and (sst-20 >= 0) and (sst+21 < dim):
                    y3[i, sst-20:sst+21, 0] = _label()
                    y3[len(list_IDs)+i, sst-20:sst+21, 0] = _label()                   
                elif sst and (sst+21 < dim):
                    y3[i, 0:sst+sst+1, 0] = _label(a=0, b=sst, c=2*sst)
                    y3[len(list_IDs)+i, 0:sst+sst+1, 0] = _label(a=0, b=sst, c=2*sst)
                elif sst and (sst-20 >= 0):
                    sdif = dim - sst
                    y3[i, sst-sdif-1:dim, 0] = _label(a=sst-sdif, b=sst, c=2*sdif)    
                    y3[len(list_IDs)+i, sst-sdif-1:dim, 0] = _label(a=sst-sdif, b=sst, c=2*sdif)    
                     
                sd = sst - spt      
                if sst+int(0.4*sd) <= dim: 
                    y1[i, spt:int(sst+(0.4*sd)), 0] = 1  
                    y1[len(list_IDs)+i, spt:int(sst+(0.4*sd)), 0] = 1                              
                else:
                    y1[i, spt:dim, 0] = 1
                    y1[len(list_IDs)+i, spt:dim, 0] = 1  
                    
                if additions: 
                    add_spt = additions[0];
                    print(add_spt)
                    add_sst = additions[1];
                    add_sd = add_sst - add_spt 
                    
                    if add_spt and (add_spt-20 >= 0) and (add_spt+21 < dim):
                        y2[len(list_IDs)+i, add_spt-20:add_spt+21, 0] = _label()  
                    elif add_spt and (add_spt+21 < dim):
                        y2[len(list_IDs)+i, 0:add_spt+add_spt+1, 0] = _label(a=0, b=add_spt, c=2*add_spt)
                    elif add_spt and (add_spt-20 >= 0):
                        pdif = dim - add_spt
                        y2[len(list_IDs)+i, add_spt-pdif-1:dim, 0] = _label(a=add_spt-pdif, b=add_spt, c=2*pdif) 
                        
                    if add_sst and (add_sst-20 >= 0) and (add_sst+21 < dim):
                        y3[len(list_IDs)+i, add_sst-20:add_sst+21, 0] = _label()
                    elif add_sst and (add_sst+21 < dim):
                        y3[len(list_IDs)+i, 0:add_sst+add_sst+1, 0] = _label(a=0, b=add_sst, c=2*add_sst)
                    elif add_sst and (add_sst-20 >= 0):
                        sdif = dim - add_sst
                        y3[len(list_IDs)+i, add_sst-sdif-1:dim, 0] = _label(a=add_sst-sdif, b=add_sst, c=2*sdif) 
    
                    if add_sst+int(0.4*add_sd) <= dim: 
                        y1[len(list_IDs)+i, add_spt:int(add_sst+(0.4*add_sd)), 0] = 1        
                    else:
                        y1[len(list_IDs)+i, add_spt:dim, 0] = 1

    fl.close()                           
    return X.astype('float32'), y1.astype('float32'), y2.astype('float32'), y3.astype('float32')




class PreLoadGeneratorTest(keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing. For testing. Pre-load version.
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 file.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32.
        Batch size.
            
    n_channels: int, default=3.
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'                
            
    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
    
    
    """  
    
    def __init__(self, 
                 list_IDs, 
                 inp_data, 
                 dim, 
                 batch_size=32, 
                 n_channels=3, 
                 norm_mode = 'std'):
       
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.inp_data = inp_data        
        self.n_channels = n_channels
        self.on_epoch_end()
        self.norm_mode = norm_mode
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def _normalize(self, data, mode='max'):   
        'Normalize waveforms in each batch'
        
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data
                       
    def __data_generation(self, list_IDs_temp):
        'readint the waveforms' 
        X = np.zeros((self.batch_size, self.dim, self.n_channels))           
        # Generate data
        for i, ID in enumerate(list_IDs_temp):            
            dataset = self.inp_data[ID]
            data = np.array(dataset) 
            data = self._normalize(data, self.norm_mode)                            
            X[i, :, :] = data                                                           
                           
        return X



class DataGeneratorTest(keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing. For testing. 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 file.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32
        Batch size.
            
    n_channels: int, default=3
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.
            
    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
    
    """   
    
    def __init__(self, 
                 list_IDs, 
                 file_name, 
                 dim, 
                 batch_size=32, 
                 n_channels=3, 
                 norm_mode = 'max'):
       
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.file_name = file_name        
        self.n_channels = n_channels
        self.on_epoch_end()
        self.norm_mode = norm_mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def normalize(self, data, mode = 'max'):  
        'Normalize waveforms in each batch'
        
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data    


    def __data_generation(self, list_IDs_temp):
        'readint the waveforms' 
        
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        fl = h5py.File(self.file_name, 'r')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            if ID.split('_')[-1] == 'EV':
                dataset = fl.get('data/'+str(ID))
                data = np.array(dataset)              
                    
            elif ID.split('_')[-1] == 'NO':
                dataset = fl.get('data/'+str(ID))
                data = np.array(dataset)
          
            if self.norm_mode:                    
                data = self.normalize(data, self.norm_mode)  
                            
            X[i, :, :] = data                                       

        fl.close() 
                           
        return X



class DataGeneratorPrediction(keras.utils.Sequence):
    
    """ 
    Keras generator with preprocessing. For prediction. 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 file.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32
        Batch size.
            
    n_channels: int, default=3
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.

        
    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
   
    
    """   
    
    def __init__(self, 
                 list_IDs, 
                 file_name, 
                 dim, 
                 batch_size=32, 
                 n_channels=3, 
                 norm_mode = 'max'):
       
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.file_name = file_name        
        self.n_channels = n_channels
        self.on_epoch_end()
        self.norm_mode = norm_mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def normalize(self, data, mode = 'max'): 
        'Normalize waveforms in a batch'
         
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data    
 
    def __data_generation(self, list_IDs_temp):
        'read the waveforms'         
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        fl = h5py.File(self.file_name, 'r')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            dataset = fl.get('data/'+str(ID))
            data = np.array(dataset)                
         
            if self.norm_mode:                    
                data = self.normalize(data, self.norm_mode)  
                            
            X[i, :, :] = data                                       

        fl.close() 
                           
        return X




def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    ---------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
   ----------------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind




def picker(args, yh1, yh2, yh3, yh1_std, yh2_std, yh3_std, spt=None, sst=None):

    """ 
    
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array
        Detection probabilities. 
        
    yh2 : 1D array
        P arrival probabilities.  
        
    yh3 : 1D array
        S arrival probabilities. 
        
    yh1_std : 1D array
        Detection standard deviations. 
        
    yh2_std : 1D array
        P arrival standard deviations.  
        
    yh3_std : 1D array
        S arrival standard deviations. 
        
    spt : {int, None}, default=None    
        P arrival time in sample.
        
    sst : {int, None}, default=None
        S arrival time in sample. 
        
   
    Returns
    --------    
    matches: dic
        Contains the information for the detected and picked event.            
        
    matches: dic
        {detection statr-time:[ detection end-time, detection probability, detectin uncertainty, P arrival, P probabiliy, P uncertainty, S arrival,  S probability, S uncertainty]}
            
    pick_errors : dic                
        {detection statr-time:[ P_ground_truth - P_pick, S_ground_truth - S_pick]}
        
    yh3: 1D array             
        normalized S_probability                              
                
    """               
    
 #   yh3[yh3>0.04] = ((yh1+yh3)/2)[yh3>0.04] 
 #   yh2[yh2>0.10] = ((yh1+yh2)/2)[yh2>0.10] 
             
    detection = trigger_onset(yh1, args['detection_threshold'], args['detection_threshold'])
    pp_arr = _detect_peaks(yh2, mph=args['P_threshold'], mpd=1)
    ss_arr = _detect_peaks(yh3, mph=args['S_threshold'], mpd=1)
          
    P_PICKS = {}
    S_PICKS = {}
    EVENTS = {}
    matches = {}
    pick_errors = {}
    if len(pp_arr) > 0:
        P_uncertainty = None  
            
        for pick in range(len(pp_arr)): 
            pauto = pp_arr[pick]
                        
            if args['estimate_uncertainty'] and pauto:
                P_uncertainty = np.round(yh2_std[int(pauto)], 3)
                    
            if pauto: 
                P_prob = np.round(yh2[int(pauto)], 3) 
                P_PICKS.update({pauto : [P_prob, P_uncertainty]})                 
                
    if len(ss_arr) > 0:
        S_uncertainty = None  
            
        for pick in range(len(ss_arr)):        
            sauto = ss_arr[pick]
                   
            if args['estimate_uncertainty'] and sauto:
                S_uncertainty = np.round(yh3_std[int(sauto)], 3)
                    
            if sauto: 
                S_prob = np.round(yh3[int(sauto)], 3) 
                S_PICKS.update({sauto : [S_prob, S_uncertainty]})             
            
    if len(detection) > 0:
        D_uncertainty = None  
        
        for ev in range(len(detection)):                                 
            if args['estimate_uncertainty']:               
                D_uncertainty = np.mean(yh1_std[detection[ev][0]:detection[ev][1]])
                D_uncertainty = np.round(D_uncertainty, 3)
                    
            D_prob = np.mean(yh1[detection[ev][0]:detection[ev][1]])
            D_prob = np.round(D_prob, 3)
                    
            EVENTS.update({ detection[ev][0] : [D_prob, D_uncertainty, detection[ev][1]]})            
    
    # matching the detection and picks
    def pair_PS(l1, l2, dist):
        l1.sort()
        l2.sort()
        b = 0
        e = 0
        ans = []
        
        for a in l1:
            while l2[b] and b < len(l2) and a - l2[b] > dist:
                b += 1
            while l2[e] and e < len(l2) and l2[e] - a <= dist:
                e += 1
            ans.extend([[a,x] for x in l2[b:e]])
            
        best_pair = None
        for pr in ans: 
            ds = pr[1]-pr[0]
            if abs(ds) < dist:
                best_pair = pr
                dist = ds           
        return best_pair


    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][2]
        S_error = None
        P_error = None        
        if int(ed-bg) >= 10:
                                    
            candidate_Ss = {}
            for Ss, S_val in S_PICKS.items():
                if Ss > bg and Ss < ed:
                    candidate_Ss.update({Ss : S_val}) 
             
            if len(candidate_Ss) > 1:                
# =============================================================================
#                 Sr_st = 0
#                 buffer = {}
#                 for SsCan, S_valCan in candidate_Ss.items():
#                     if S_valCan[0] > Sr_st:
#                         buffer = {SsCan : S_valCan}
#                         Sr_st = S_valCan[0]
#                 candidate_Ss = buffer
# =============================================================================              
                candidate_Ss = {list(candidate_Ss.keys())[0] : candidate_Ss[list(candidate_Ss.keys())[0]]}


            if len(candidate_Ss) == 0:
                    candidate_Ss = {None:[None, None]}

            candidate_Ps = {}
            for Ps, P_val in P_PICKS.items():
                if list(candidate_Ss)[0]:
                    if Ps > bg-100 and Ps < list(candidate_Ss)[0]-10:
                        candidate_Ps.update({Ps : P_val}) 
                else:         
                    if Ps > bg-100 and Ps < ed:
                        candidate_Ps.update({Ps : P_val}) 
                    
            if len(candidate_Ps) > 1:
                Pr_st = 0
                buffer = {}
                for PsCan, P_valCan in candidate_Ps.items():
                    if P_valCan[0] > Pr_st:
                        buffer = {PsCan : P_valCan} 
                        Pr_st = P_valCan[0]
                candidate_Ps = buffer
                    
            if len(candidate_Ps) == 0:
                    candidate_Ps = {None:[None, None]}
                    
                    
# =============================================================================
#             Ses =[]; Pes=[]
#             if len(candidate_Ss) >= 1:
#                 for SsCan, S_valCan in candidate_Ss.items():
#                     Ses.append(SsCan) 
#                                 
#             if len(candidate_Ps) >= 1:
#                 for PsCan, P_valCan in candidate_Ps.items():
#                     Pes.append(PsCan) 
#             
#             if len(Ses) >=1 and len(Pes) >= 1:
#                 PS = pair_PS(Pes, Ses, ed-bg)
#                 if PS:
#                     candidate_Ps = {PS[0] : candidate_Ps.get(PS[0])}
#                     candidate_Ss = {PS[1] : candidate_Ss.get(PS[1])}
# =============================================================================

            if list(candidate_Ss)[0] or list(candidate_Ps)[0]:                 
                matches.update({
                                bg:[ed, 
                                    EVENTS[ev][0], 
                                    EVENTS[ev][1], 
                                
                                    list(candidate_Ps)[0],  
                                    candidate_Ps[list(candidate_Ps)[0]][0], 
                                    candidate_Ps[list(candidate_Ps)[0]][1],  
                                                
                                    list(candidate_Ss)[0],  
                                    candidate_Ss[list(candidate_Ss)[0]][0], 
                                    candidate_Ss[list(candidate_Ss)[0]][1],  
                                                ] })
                
                if sst and sst > bg and sst < EVENTS[ev][2]:
                    if list(candidate_Ss)[0]:
                        S_error = sst -list(candidate_Ss)[0] 
                    else:
                        S_error = None
                                            
                if spt and spt > bg-100 and spt < EVENTS[ev][2]:
                    if list(candidate_Ps)[0]:  
                        P_error = spt - list(candidate_Ps)[0] 
                    else:
                        P_error = None
                                          
                pick_errors.update({bg:[P_error, S_error]})
      
    return matches, pick_errors, yh3





def generate_arrays_from_file(file_list, step):
    
    """ 
    
    Make a generator to generate list of trace names.
    
    Parameters
    ----------
    file_list : str
        A list of trace names.  
        
    step : int
        Batch size.  
        
    Returns
    --------  
    chunck : str
        A batch of trace names. 
        
    """     
    
    n_loops = int(np.ceil(len(file_list) / step))
    b = 0
    while True:
        for i in range(n_loops):
            e = i*step + step 
            if e > len(file_list):
                e = len(file_list)
            chunck = file_list[b:e]
            b=e
            yield chunck   

    
    

def f1(y_true, y_pred):
    
    """ 
    
    Calculate F1-score.
    
    Parameters
    ----------
    y_true : 1D array
        Ground truth labels. 
        
    y_pred : 1D array
        Predicted labels.     
        
    Returns
    -------  
    f1 : float
        Calculated F1-score. 
        
    """     
    
    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        'Precision metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def normalize(data, mode='std'):
    
    """ 
    
    Normalize 3D arrays.
    
    Parameters
    ----------
    data : 3D numpy array
        3 component traces. 
        
    mode : str, default='std'
        Mode of normalization. 'max' or 'std'     
        
    Returns
    -------  
    data : 3D numpy array
        normalized data. 
            
    """   
    
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data              
    elif mode == 'std':        
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data
    
    

  
class LayerNormalization(keras.layers.Layer):
    
    """ 
    
    Layer normalization layer modified from https://github.com/CyberZHG based on [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
    
    Parameters
    ----------
    center: bool
        Add an offset parameter if it is True. 
        
    scale: bool
        Add a scale parameter if it is True.     
        
    epsilon: bool
        Epsilon for calculating variance.     
        
    gamma_initializer: str
        Initializer for the gamma weight.     
        
    beta_initializer: str
        Initializer for the beta weight.     
                    
    Returns
    -------  
    data: 3D tensor
        with shape: (batch_size, …, input_dim) 
            
    """   
              
    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 **kwargs):

        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
      

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = keras.engine.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

    
    
class FeedForward(keras.layers.Layer):
    """Position-wise feed-forward layer. modified from https://github.com/CyberZHG 
    # Arguments
        units: int >= 0. Dimension of hidden units.
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.
    # Input shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # Output shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # References
        - [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """
    
    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 dropout_rate=0.0,
                 **kwargs):
        self.supports_masking = True
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.dropout_rate = dropout_rate
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='{}_b1'.format(self.name),
            )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        h = K.dot(x, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        if 0.0 < self.dropout_rate < 1.0:
            def dropped_inputs():
                return K.dropout(h, self.dropout_rate, K.shape(h))
            h = K.in_train_phase(dropped_inputs, h, training=training)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y


class SeqSelfAttention(keras.layers.Layer):
    """Layer initialization. modified from https://github.com/CyberZHG
    For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf
    :param units: The dimension of the vectors that used to calculate the attention weights.
    :param attention_width: The width of local attention.
    :param attention_type: 'additive' or 'multiplicative'.
    :param return_attention: Whether to return the attention weights for visualization.
    :param history_only: Only use historical pieces of data.
    :param kernel_initializer: The initializer for weight matrices.
    :param bias_initializer: The initializer for biases.
    :param kernel_regularizer: The regularization for weight matrices.
    :param bias_regularizer: The regularization for biases.
    :param kernel_constraint: The constraint for weight matrices.
    :param bias_constraint: The constraint for biases.
    :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                              in additive mode.
    :param use_attention_bias: Whether to use bias while calculating the weights of attention.
    :param attention_activation: The activation used for calculating the weights of attention.
    :param attention_regularizer_weight: The weights of attention regularizer.
    :param kwargs: Parameters for parent class.
    """
        
    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):

        super(SeqSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': keras.regularizers.serialize(self.kernel_initializer),
            'bias_initializer': keras.regularizers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'attention_activation': keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        input_len = K.shape(inputs)[1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(0, input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(0, input_len), axis=0)
            e = e * K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx())
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            e = K.permute_dimensions(K.permute_dimensions(e * mask, (0, 2, 1)) * mask, (0, 2, 1))

        # a_{t} = \text{softmax}(e_t)
        s = K.sum(e, axis=-1, keepdims=True)
        a = e / (s + K.epsilon())

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}



def _block_BiLSTM(filters, drop_rate, padding, inpR):
    'Returns LSTM residual block'    
    prev = inpR
    x_rnn = Bidirectional(LSTM(filters, return_sequences=True, dropout=drop_rate, recurrent_dropout=drop_rate))(prev)
    NiN = Conv1D(filters, 1, padding = padding)(x_rnn)     
    res_out = BatchNormalization()(NiN)
    return res_out


def _block_CNN_1(filters, ker, drop_rate, activation, padding, inpC): 
    ' Returns CNN residual blocks '
    prev = inpC
    layer_1 = BatchNormalization()(prev) 
    act_1 = Activation(activation)(layer_1) 
    act_1 = SpatialDropout1D(drop_rate)(act_1, training=True)
    conv_1 = Conv1D(filters, ker, padding = padding)(act_1) 
    
    layer_2 = BatchNormalization()(conv_1) 
    act_2 = Activation(activation)(layer_2) 
    act_2 = SpatialDropout1D(drop_rate)(act_2, training=True)
    conv_2 = Conv1D(filters, ker, padding = padding)(act_2)
    
    res_out = add([prev, conv_2])
    
    return res_out 


def _transformer(drop_rate, width, name, inpC): 
    ' Returns a transformer block containing one addetive attention and one feed  forward layer with residual connections '
    x = inpC
    
    att_layer, weight = SeqSelfAttention(return_attention =True,                                       
                                         attention_width = width,
                                         name=name)(x)
   
#  att_layer = Dropout(drop_rate)(att_layer, training=True)    
    att_layer2 = add([x, att_layer])    
    norm_layer = LayerNormalization()(att_layer2)
    
    FF = FeedForward(units=128, dropout_rate=drop_rate)(norm_layer)
    
    FF_add = add([norm_layer, FF])    
    norm_out = LayerNormalization()(FF_add)
    
    return norm_out, weight 

     

def _encoder(filter_number, filter_size, depth, drop_rate, ker_regul, bias_regul, activation, padding, inpC):
    ' Returns the encoder that is a combination of residual blocks and maxpooling.'        
    e = inpC
    for dp in range(depth):
        e = Conv1D(filter_number[dp], 
                   filter_size[dp], 
                   padding = padding, 
                   activation = activation,
                   kernel_regularizer = ker_regul,
                   bias_regularizer = bias_regul,
                   )(e)             
        e = MaxPooling1D(2, padding = padding)(e)            
    return(e) 


def _decoder(filter_number, filter_size, depth, drop_rate, ker_regul, bias_regul, activation, padding, inpC):
    ' Returns the dencoder that is a combination of residual blocks and upsampling. '           
    d = inpC
    for dp in range(depth):        
        d = UpSampling1D(2)(d) 
        if dp == 3:
            d = Cropping1D(cropping=(1, 1))(d)           
        d = Conv1D(filter_number[dp], 
                   filter_size[dp], 
                   padding = padding, 
                   activation = activation,
                   kernel_regularizer = ker_regul,
                   bias_regularizer = bias_regul,
                   )(d)        
    return(d)  
 


def _lr_schedule(epoch):
    ' Learning rate is scheduled to be reduced after 40, 60, 80, 90 epochs.'
    
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr



class cred2():
    
    """ 
    
    Creates the model
    
    Parameters
    ----------
    nb_filters: list
        The list of filter numbers. 
        
    kernel_size: list
        The size of the kernel to use in each convolutional layer.
        
    padding: str
        The padding to use in the convolutional layers.

    activationf: str
        Activation funciton type.

    endcoder_depth: int
        The number of layers in the encoder.
        
    decoder_depth: int
        The number of layers in the decoder.

    cnn_blocks: int
        The number of residual CNN blocks.

    BiLSTM_blocks: int=
        The number of Bidirectional LSTM blocks.
  
    drop_rate: float 
        Dropout rate.

    loss_weights: list
        Weights of the loss function for the detection, P picking, and S picking.       
                
    loss_types: list
        Types of the loss function for the detection, P picking, and S picking. 

    kernel_regularizer: str
        l1 norm regularizer.

    bias_regularizer: str
        l1 norm regularizer.

    multi_gpu: bool
        If use multiple GPUs for the training. 

    gpu_number: int
        The number of GPUs for the muli-GPU training. 
           
    Returns
    ----------
        The complied model: keras model
        
    """

    def __init__(self,
                 nb_filters=[8, 16, 16, 32, 32, 96, 96, 128],
                 kernel_size=[11, 9, 7, 7, 5, 5, 3, 3],
                 padding='same',
                 activationf='relu',
                 endcoder_depth=7,
                 decoder_depth=7,
                 cnn_blocks=5,
                 BiLSTM_blocks=3,
                 drop_rate=0.1,
                 loss_weights=[0.2, 0.3, 0.5],
                 loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],                                 
                 kernel_regularizer=keras.regularizers.l1(1e-4),
                 bias_regularizer=keras.regularizers.l1(1e-4),
                 multi_gpu=False, 
                 gpu_number=4, 
                 ):
        
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.activationf = activationf
        self.endcoder_depth= endcoder_depth
        self.decoder_depth= decoder_depth
        self.cnn_blocks= cnn_blocks
        self.BiLSTM_blocks= BiLSTM_blocks     
        self.drop_rate= drop_rate
        self.loss_weights= loss_weights  
        self.loss_types = loss_types       
        self.kernel_regularizer = kernel_regularizer     
        self.bias_regularizer = bias_regularizer 
        self.multi_gpu = multi_gpu
        self.gpu_number = gpu_number

        
    def __call__(self, inp):

        x = inp
        x = _encoder(self.nb_filters, 
                    self.kernel_size, 
                    self.endcoder_depth, 
                    self.drop_rate, 
                    self.kernel_regularizer, 
                    self.bias_regularizer,
                    self.activationf, 
                    self.padding,
                    x)    
        
        for cb in range(self.cnn_blocks):
            x = _block_CNN_1(self.nb_filters[6], 3, self.drop_rate, self.activationf, self.padding, x)
            if cb > 2:
                x = _block_CNN_1(self.nb_filters[6], 2, self.drop_rate, self.activationf, self.padding, x)

        for bb in range(self.BiLSTM_blocks):
            x = _block_BiLSTM(self.nb_filters[1], self.drop_rate, self.padding, x)

            
        x, weightdD0 = _transformer(self.drop_rate, None, 'attentionD0', x)             
        encoded, weightdD = _transformer(self.drop_rate, None, 'attentionD', x)             
            
        decoder_D = _decoder([i for i in reversed(self.nb_filters)], 
                             [i for i in reversed(self.kernel_size)], 
                             self.decoder_depth, 
                             self.drop_rate, 
                             self.kernel_regularizer, 
                             self.bias_regularizer,
                             self.activationf, 
                             self.padding,                             
                             encoded)
        d = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='detector')(decoder_D)


        PLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded)
        norm_layerP, weightdP = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionP')(PLSTM)
        
        decoder_P = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)], 
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerP)
        P = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_P')(decoder_P)
        
        SLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded) 
        norm_layerS, weightdS = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionS')(SLSTM)
        
        
        decoder_S = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)],
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerS) 
        
        S = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_S')(decoder_S)
        

        if self.multi_gpu == True:
            parallel_model = Model(inputs=inp, outputs=[d, P, S])
            model = multi_gpu_model(parallel_model, gpus=self.gpu_number)
        else:
            model = Model(inputs=inp, outputs=[d, P, S])

        model.compile(loss=self.loss_types, loss_weights=self.loss_weights,    
            optimizer=Adam(lr=_lr_schedule(0)), metrics=[f1])

        return model

        


