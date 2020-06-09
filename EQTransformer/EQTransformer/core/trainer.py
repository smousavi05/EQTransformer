#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:44:14 2018

@author: mostafamousavi
last update: 06/06/2020

"""

from __future__ import print_function
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import time
import os
import shutil
import multiprocessing
from .EqT_utils import DataGenerator, _lr_schedule, cred2, PreLoadGenerator, data_reader
import datetime
from tqdm import tqdm
import argparse
 

def trainer(input_hdf5=None,
            input_csv=None,
            output_name=None,                
            input_dimention=(6000, 3),
            cnn_blocks=5,
            lstm_blocks=2,
            padding='same',
            activation = 'relu',            
            drop_rate=0.2,
            shuffle=True, 
            label_type='gaussian',
            normalization_mode='std',
            augmentation=True,
            add_event_r=0.6,
            shift_event_r=0.99,
            add_noise_r=0.3, 
            drop_channel_r=0.5,
            add_gap_r=0.2,
            scale_amplitude_r=None,
            pre_emphasis=False,                
            loss_weights=[0.05, 0.40, 0.55],
            loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
            train_valid_test_split=[0.85, 0.05, 0.10],
            mode='generator',
            batch_size=200,
            epochs=200, 
            monitor='val_loss',
            patience=12,
            multi_gpu=False,
            number_of_gpus=4,
            gpuid=None,
            gpu_limit=None,
            use_multiprocessing=True):
        
    """
    Generate a model and train it. 

    Parameters
    ----------
       input_hdf5: str, (default = None)
           Path to s hdf5 file containing only one class of "data" with numpy arries containing 3 component waveforms each 1 min long.
           
       input_csv: str, (default = None)
           Path to a csv file with one column (trace_name) listing the name of all datasets in the hdf5 file.
           
       output_name: str, (default = None)
           Output directory.
           
       input_dimention: tuple, (default = (6000, 3)) 
           Dimension of input traces. 
           
       cnn_blocks: positive integer, (default = 5) 
           Number of residual blocks of convolutional layers.   
           
       lstm_blocks: positive integer, (default = 2) 
           Number of residual blocks of BiLSTM layers.  

       padding: str, (default = 'same') 
           Padding type.   
           
       activation: str, (default = 'relu') 
           Activation function used in the hidden layers.             
           
       drop_rate: float, (default = 0.2) 
           Droupout value.
           
       shuffle: bool, (default = True) 
           To shuffle the list prior to the training. 
           
       label_type: str, (default = 'triangle') 
           Labeling type. 'gaussian', 'triangle', or 'box' 
           
       normalization_mode: str, (default = 'std') 
           Mode of normalization for data preprocessing, 'max': maximum amplitude among three componenet, 'std': standard deviation.  
           
       augmentation: bool, (default = True)
           If True, data will be augmented simultaneously during the training.  
           
       add_event_r: {float, None}, (default = 0.6)
           Rate of augmentation for adding a secondary event randomly into the empty part of a trace. 
           
       shift_event_r: {float, None}, (default = 0.99)
           Rate of augmentation for randomly shifting the event within a trace.    
           
       add_noise_r: {float, None}, (default = 0.3)
           Rate of augmentation for adding Gaussian noise with different SNR into a trace. 
           
       drop_channel_r: {float, None}, (default = 0.4)
           Rate of augmentation for randomly dropping one of the channels. 

        add_gap_r : {float, None}, (default=0.2)
            Add an interval with zeros into the waveform representing filled gaps.
            
       scale_amplitude_r: {float, None}, (default = None)
           Rate of augmentation for randomly scaling the trace. 
           
       pre_emphasis: bool, (default = True)
           If True, waveforms will be pre emphasized. 
           
       loss_weights: list of three floats, (default = [0.05, 0.40, 0.55])
           Loss wieghts for detection, P picking, and S picking respectively.
           
       loss_types: list of three str, (default = ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'])
           Loss types for detection, P picking, and S picking respectively.     
           
       train_valid_test_split: list of three floats, (default = [0.85, 0.05, 0.10])
           Precentage of data split into the training, validation, and test sets respectively.
           
       mode : str, (default = 'generator')
           Mode of running. 'pre_load_generator',  'generator', or 'preload'.
           
       batch_size: positive integer, (default = 200) 
           Batch size.             
           
       epochs: positive integer, (default = 200) 
           Number of epochs. 
           
       monitor: str, (default = 'val_loss') 
           Measure used for the monitoring.  
           
       patience: positive integer, (default = 12) 
           Number of epochs without any improvement in the monitoring measure to automatically stop the training.
           
       multi_gpu: bool, (default = False)
           If True, multiple GPUs will be used for the training. 
           
       number_of_gpus: positive integer, (default = 4)
           Number of GPUs uses for multi-gpu training.  
           
       gpuid: {positive integer, None}, (default = None) 
           Id of GPU used for the prediction. If using CPU set to None. 
           
       gpu_limit: {positive integer, None}, (default = None) 
           Set the maximum precentage of memomry usage for the GPU. 
           
       use_multiprocessing: bool, (default = True) 
           If True, multiple CPUs will be used for the preprocessing of data even when GPU is used for the prediction.

           
    rubric:: Notes
        'pre_load' and 'pre_load_generator' modes are faster than 'generator' but requiers much more memory.
        They are appropriate for the machines with tones of memory but slow disks. 
        'generator' on the other hand, is memory efficient and more suitable for machines with fast disks. 
        'pre_load' mode comes with only with box labeling.

    Generates
    -------
    ./output_name/models/output_name_???.h5
        This is where all good models will be saved.
        
    ./output_name/final_model.h5  
        This is the full model for the last epoch.
        
    ./output_name/model_weights.h5  
        This is the wieghts for the last model.
        
    ./output_name/history.npy  
        Training history.    
        
    ./output_name/X_report.txt
        A summary of parameters used for the prediction and perfomance.
        
    ./output_name/test.npy
        A numpy list containing the trace names for the test set. 
        
    ./output_name/X_learning_curve_f1.png
        The learning curve of Fi-scores.      
        
    ./output_name/X_learning_curve_loss.png
        The learning curve of loss.      
                   
    """            
                           
    parser = argparse.ArgumentParser()    
    parser.add_argument("--input_hdf5", default= input_hdf5)
    parser.add_argument("--input_csv", default=input_csv) 
    parser.add_argument("--output_name", default=output_name)
    parser.add_argument("--input_dimention", default=input_dimention)
    parser.add_argument("--cnn_blocks", default=cnn_blocks)  
    parser.add_argument("--lstm_blocks", default=lstm_blocks)  
    parser.add_argument("--padding", default=padding)  
    parser.add_argument("--activation", default=activation)     
    parser.add_argument("--drop_rate", default=drop_rate)  
    parser.add_argument("--shuffle", default=shuffle)  
    parser.add_argument("--label_type", default=label_type)  
    parser.add_argument("--normalization_mode", default=normalization_mode)  
    parser.add_argument("--augmentation", default=augmentation)  
    parser.add_argument("--add_event_r", default=add_event_r)  
    parser.add_argument("--shift_event_r", default=shift_event_r)  
    parser.add_argument("--add_noise_r", default=add_noise_r)  
    parser.add_argument("--add_gap_r", default=add_gap_r)      
    parser.add_argument("--drop_channel_r", default=drop_channel_r)  
    parser.add_argument("--scale_amplitude_r", default=scale_amplitude_r)  
    parser.add_argument("--pre_emphasis", default=pre_emphasis)  
    parser.add_argument("--loss_weights", default=loss_weights)  
    parser.add_argument("--loss_types", default=loss_types)  
    parser.add_argument("--train_valid_test_split", default=train_valid_test_split)  
    parser.add_argument("--mode", default=mode)  
    parser.add_argument("--batch_size", default=batch_size)  
    parser.add_argument("--epochs", default=epochs)  
    parser.add_argument("--monitor", default=monitor)  
    parser.add_argument("--patience", default=patience)  
    parser.add_argument("--multi_gpu", default=multi_gpu)  
    parser.add_argument("--number_of_gpus", default=number_of_gpus)  
    parser.add_argument("--gpuid", default=gpuid)  
    parser.add_argument("--gpu_limit", default=gpu_limit)  
    parser.add_argument("--use_multiprocessing", default=use_multiprocessing)   
    args = parser.parse_args()

                       
    def train(ar):
        """ 
        Performs the training.
    
        Parameters
        ----------
        args : object
            A argparse object containing all of the input parameters. 

        Returns
        -------
        history : dic
            Training history.     
        model : 
            Trained model.  
        start_training : datetime
            Training start time. 
        end_training : datetime
            Training end time.             
        save_dir : str
            Path to the output directory. 
        save_models : str
            Path to the folder for saveing the models.        
        training size : int
            Number of training samples.    
        validation size : int
            Number of validation samples.              
        """    

        
        save_dir, save_models=_make_dir(ar.output_name)
        training, validation=_split(ar, save_dir)
        callbacks=_make_callback(ar, save_models)
        model=_build_model(ar)
        
        if ar.gpuid:           
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpuid)
            tf.Session(config=tf.ConfigProto(log_device_placement=True))
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = float(ar.gpu_limit) 
            K.tensorflow_backend.set_session(tf.Session(config=config))
            
        start_training = time.time()                  
        if ar.mode == 'pre_load_generator': 
            
            if ar.input_hdf5 == None:
                print('Please specify the input_hdf5!')
                return
            else:
                training_generator, validation_generator=_pre_loading(ar, training, validation)

                print('Started training in pre_load_generator mode ...', flush=True) 
                history = model.fit_generator(generator=training_generator,
                                              validation_data=validation_generator,
                                              use_multiprocessing=False,                         
                                              callbacks=callbacks, 
                                              epochs=ar.epochs,
                                              class_weight={0: 0.11, 1: 0.89})
            
        elif ar.mode == 'generator': 
            
            params_training = {'file_name': str(ar.input_hdf5), 
                              'dim': ar.input_dimention[0],
                              'batch_size': ar.batch_size,
                              'n_channels': ar.input_dimention[-1],
                              'shuffle': ar.shuffle,  
                              'norm_mode': ar.normalization_mode,
                              'label_type': ar.label_type,                          
                              'augmentation': ar.augmentation,
                              'add_event_r': ar.add_event_r, 
                              'add_gap_r': ar.add_gap_r,  
                              'shift_event_r': ar.shift_event_r,                            
                              'add_noise_r': ar.add_noise_r, 
                              'drop_channe_r': ar.drop_channel_r,
                              'scale_amplitude_r': ar.scale_amplitude_r,
                              'pre_emphasis': ar.pre_emphasis}    
                        
            params_validation = {'file_name': str(ar.input_hdf5),  
                                 'dim': ar.input_dimention[0],
                                 'batch_size': ar.batch_size,
                                 'n_channels': ar.input_dimention[-1],
                                 'shuffle': False,  
                                 'norm_mode': ar.normalization_mode,
                                 'augmentation': False}         

            training_generator = DataGenerator(training, **params_training)
            validation_generator = DataGenerator(validation, **params_validation) 

            print('Started training in generator mode ...') 
            history = model.fit_generator(generator=training_generator,
                                          validation_data=validation_generator,
                                          use_multiprocessing=ar.use_multiprocessing,
                                          workers=multiprocessing.cpu_count(),    
                                          callbacks=callbacks, 
                                          epochs=ar.epochs,
                                          class_weight={0: 0.11, 1: 0.89})
            
        elif ar.mode == 'preload': 
            X, y1, y2, y3 = data_reader(list_IDs=training+validation, 
                                       file_name=str(ar.input_hdf5), 
                                       dim=ar.input_dimention[0], 
                                       n_channels=ar.input_dimention[-1], 
                                       norm_mode=ar.normalization_mode,
                                       augmentation=ar.augmentation,
                                       add_event_r=ar.add_event_r,
                                       add_gap_r=ar.add_gap_r, 
                                       shift_event_r=ar.shift_event_r, 
                                       add_noise_r=ar.add_noise_r,  
                                       drop_channe_r=ar.drop_channel_r,
                                       scale_amplitude_r=ar.scale_amplitude_r,
                                       pre_emphasis=ar.pre_emphasis)
             
            print('Started training in preload mode ...', flush=True) 
            history = model.fit({'input': X}, 
                                {'detector': y1, 'picker_P': y2, 'picker_S': y3}, 
                                epochs=ar.epochs,
                                validation_split=ar.train_valid_test_split[1],
                                batch_size=ar.batch_size, 
                                callbacks=callbacks,
                                class_weight={0: 0.11, 1: 0.89})            
        else:
            print('Please specify training_mode !', flush=True)
        end_training = time.time()  
        
        return history, model, start_training, end_training, save_dir, save_models, len(training), len(validation)
                  
    history, model, start_training, end_training, save_dir, save_models, training_size, validation_size=train(args)  
    _document_training(history, model, start_training, end_training, save_dir, save_models, training_size, validation_size, args)





def _make_dir(output_name):
    """ 
    Make the output directories.

    Parameters
    ----------
    output_name : str
        Name of the output directory.
                   
    Returns
    -------   
    save_dir : str
        Full path to the output directory.
    save_models : str
        Full path to the model directory.        
    """   
    
    if output_name == None:
        print('Please specify output_name!') 
        return
    else:
        save_dir = os.path.join(os.getcwd(), str(output_name)+'_outputs')
        save_models = os.path.join(save_dir, 'models')      
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)  
        os.makedirs(save_models)
    return save_dir, save_models



def _build_model(ar):   
    """ 
    Build and compile the model.

    Parameters
    ----------
    ar : object
        A argparse object containing all of the input parameters. 
               
    Returns
    -------   
    model : 
        Compiled model.  
    """       
    
    inp = Input(shape=ar.input_dimention, name='input') 
    model = cred2(nb_filters=[8, 16, 16, 32, 32, 64, 64],
              kernel_size=[11, 9, 7, 7, 5, 5, 3],
              padding=ar.padding,
              activationf =ar.activation,
              cnn_blocks=ar.cnn_blocks,
              BiLSTM_blocks=ar.lstm_blocks,
              drop_rate=ar.drop_rate, 
              loss_weights=ar.loss_weights,
              loss_types=ar.loss_types,
              kernel_regularizer=keras.regularizers.l2(1e-6),
              bias_regularizer=keras.regularizers.l1(1e-4),
              multi_gpu=ar.multi_gpu, 
              gpu_number=ar.number_of_gpus,  
               )(inp)  
    model.summary()  
    return model  
    


def _split(ar, save_dir):
    
    """ 
    Split the list of input data into training, validation, and test set.

    Parameters
    ----------
    ar : object
        A argparse object containing all of the input parameters. 
    save_dir : str
       Path to the output directory. 
              
    Returns
    -------   
    training : str
        List of trace names for the training set. 
    validation : str
        List of trace names for the validation set. 
        
    Generates
    -------   
    test.npy : str
        List of trace names for the test set.     
        
    """       
    
    df = pd.read_csv(ar.input_csv)
    ev_list = df.trace_name.tolist()    
    np.random.shuffle(ev_list)     
    training = ev_list[:int(ar.train_valid_test_split[0]*len(ev_list))]
    validation =  ev_list[int(ar.train_valid_test_split[0]*len(ev_list)):
                            int(ar.train_valid_test_split[0]*len(ev_list) + ar.train_valid_test_split[1]*len(ev_list))]
    test =  ev_list[ int(ar.train_valid_test_split[0]*len(ev_list) + ar.train_valid_test_split[1]*len(ev_list)):]
    np.save(save_dir+'/test', test)  
    return training, validation 



def _make_callback(ar, save_models):
    
    """ 
    Generate the cullback.

    Parameters
    ----------
    ar : object
        A argparse object containing all of the input parameters. 
    save_models : str
       Path to the output directory for the models. 
              
    Returns
    -------   
    callbacks : obj
        List of callback objects. 
        
    """    
    m_name=str(ar.output_name)+'_{epoch:03d}.h5'   
    filepath=os.path.join(save_models, m_name)  
    early_stopping_monitor=EarlyStopping(monitor=ar.monitor, 
                                           patience=ar.patience) 
    checkpoint=ModelCheckpoint(filepath=filepath,
                                 monitor=ar.monitor, 
                                 mode='auto',
                                 verbose=1,
                                 save_best_only=True)  
    lr_scheduler=LearningRateScheduler(_lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=ar.patience-2,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping_monitor]
    return callbacks
 
    


def _pre_loading(ar, training, validation):
    
    """ 
    Load data into memory.

    Parameters
    ----------
    ar : object
        A argparse object containing all of the input parameters. 
    training : str
        List of trace names for the training set. 
    validation : str
        List of trace names for the validation set. 
              
    Returns
    -------   
    training_generator : obj
        Keras generator for the training set. 
    validation_generator : obj
        Keras generator for the validation set. 
        
    """   
    
    training_set={}
    fl = h5py.File(ar.input_hdf5, 'r')   
    
    print('Loading the training data into the memory ...')
    pbar = tqdm(total=len(training)) 
    for ID in training:
        pbar.update()
        if ID.split('_')[-1] == 'EV':
            dataset = fl.get('earthquake/local/'+str(ID))
        elif ID.split('_')[-1] == 'NO':
            dataset = fl.get('non_earthquake/noise/'+str(ID))
        training_set.update( {str(ID) : dataset})  

    print('Loading the validation data into the memory ...', flush=True)            
    validation_set={}
    pbar = tqdm(total=len(validation)) 
    for ID in validation:
        pbar.update()
        if ID.split('_')[-1] == 'EV':
            dataset = fl.get('earthquake/local/'+str(ID))
        elif ID.split('_')[-1] == 'NO':
            dataset = fl.get('non_earthquake/noise/'+str(ID))
        validation_set.update( {str(ID) : dataset})  
   
    params_training = {'dim':ar.input_dimention[0],
                       'batch_size': ar.batch_size,
                       'n_channels': ar.input_dimention[-1],
                       'shuffle': ar.shuffle,  
                       'norm_mode': ar.normalization_mode,
                       'label_type': ar.label_type,
                       'augmentation': ar.augmentation,
                       'add_event_r': ar.add_event_r, 
                       'add_gap_r': ar.add_gap_r,                         
                       'shift_event_r': ar.shift_event_r,  
                       'add_noise_r': ar.add_noise_r, 
                       'drop_channe_r': ar.drop_channel_r,
                       'scale_amplitude_r': ar.scale_amplitude_r,
                       'pre_emphasis': ar.pre_emphasis}  

    params_validation = {'dim': ar.input_dimention[0],
                         'batch_size': ar.batch_size,
                         'n_channels': ar.input_dimention[-1],
                         'shuffle': False,  
                         'norm_mode': ar.normalization_mode,
                         'augmentation': False}  
    
    training_generator = PreLoadGenerator(training, training_set, **params_training)  
    validation_generator = PreLoadGenerator(validation, validation_set, **params_validation) 
    
    return training_generator, validation_generator  




def _document_training(history, model, start_training, end_training, save_dir, save_models, training_size, validation_size, ar): 

    """ 
    Write down the training results.

    Parameters
    ----------
    history : dic
        Training history.     
    model : 
        Trained model.  
    start_training : datetime
        Training start time. 
    end_training : datetime
        Training end time.             
    save_dir : str
        Path to the output directory. 
    save_models : str
        Path to the folder for saveing the models.        
    training_size : int
        Number of training samples.    
    validation_size : int
        Number of validation samples. 
    ar : object
        A argparse object containing all of the input parameters. 
              
    Generates
    -------
    ./output_name/history.npy  
        Training history.    
    ./output_name/X_report.txt
        A summary of parameters used for the prediction and perfomance.
    ./output_name/X_learning_curve_f1.png
        The learning curve of Fi-scores.         
    ./output_name/X_learning_curve_loss.png
        The learning curve of loss.  
        
    """   
    
    np.save(save_dir+'/history',history)
    model.save(save_dir+'/final_model.h5')
    model.to_json()   
    model.save_weights(save_dir+'/model_weights.h5')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.plot(history.history['detector_loss'])
    ax.plot(history.history['picker_P_loss'])
    ax.plot(history.history['picker_S_loss'])
    try:
        ax.plot(history.history['val_loss'], '--')
        ax.plot(history.history['val_detector_loss'], '--')
        ax.plot(history.history['val_picker_P_loss'], '--')
        ax.plot(history.history['val_picker_S_loss'], '--') 
        ax.legend(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss', 
               'val_loss', 'val_detector_loss', 'val_picker_P_loss', 'val_picker_S_loss'], loc='upper right')
    except Exception:
        ax.legend(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss'], loc='upper right')  
        
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_loss.png'))) 
       
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['detector_f1'])
    ax.plot(history.history['picker_P_f1'])
    ax.plot(history.history['picker_S_f1'])
    try:
        ax.plot(history.history['val_detector_f1'], '--')
        ax.plot(history.history['val_picker_P_f1'], '--')
        ax.plot(history.history['val_picker_S_f1'], '--')
        ax.legend(['detector_f1', 'picker_P_f1', 'picker_S_f1', 'val_detector_f1', 'val_picker_P_f1', 'val_picker_S_f1'], loc='lower right')
    except Exception:
        ax.legend(['detector_f1', 'picker_P_f1', 'picker_S_f1'], loc='lower right')        
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_f1.png'))) 

    delta = end_training - start_training
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta    
    
    trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in model.non_trainable_weights]))
    
    with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file: 
        the_file.write('================== Overal Info =============================='+'\n')               
        the_file.write('date of report: '+str(datetime.datetime.now())+'\n')         
        the_file.write('input_hdf5: '+str(ar.input_hdf5)+'\n')            
        the_file.write('input_csv: '+str(ar.input_csv)+'\n')
        the_file.write('output_name: '+str(ar.output_name+'_outputs')+'\n')  
        the_file.write('================== Model Parameters ========================='+'\n')   
        the_file.write('input_dimention: '+str(ar.input_dimention)+'\n')
        the_file.write('cnn_blocks: '+str(ar.cnn_blocks)+'\n')
        the_file.write('lstm_blocks: '+str(ar.lstm_blocks)+'\n')
        the_file.write('padding_type: '+str(ar.padding)+'\n')
        the_file.write('activation_type: '+str(ar.activation)+'\n')        
        the_file.write('drop_rate: '+str(ar.drop_rate)+'\n')            
        the_file.write(str('total params: {:,}'.format(trainable_count + non_trainable_count))+'\n')    
        the_file.write(str('trainable params: {:,}'.format(trainable_count))+'\n')    
        the_file.write(str('non-trainable params: {:,}'.format(non_trainable_count))+'\n') 
        the_file.write('================== Training Parameters ======================'+'\n')  
        the_file.write('mode of training: '+str(ar.mode)+'\n')   
        the_file.write('loss_types: '+str(ar.loss_types)+'\n')
        the_file.write('loss_weights: '+str(ar.loss_weights)+'\n')
        the_file.write('batch_size: '+str(ar.batch_size)+'\n')
        the_file.write('epochs: '+str(ar.epochs)+'\n')   
        the_file.write('train_valid_test_split: '+str(ar.train_valid_test_split)+'\n')           
        the_file.write('total number of training: '+str(training_size)+'\n')
        the_file.write('total number of validation: '+str(validation_size)+'\n')
        the_file.write('monitor: '+str(ar.monitor)+'\n')
        the_file.write('patience: '+str(ar.patience)+'\n') 
        the_file.write('multi_gpu: '+str(ar.multi_gpu)+'\n')
        the_file.write('number_of_gpus: '+str(ar.number_of_gpus)+'\n') 
        the_file.write('gpuid: '+str(ar.gpuid)+'\n')
        the_file.write('gpu_limit: '+str(ar.gpu_limit)+'\n')             
        the_file.write('use_multiprocessing: '+str(ar.use_multiprocessing)+'\n')  
        the_file.write('================== Training Performance ====================='+'\n')  
        the_file.write('finished the training in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds,2)))                         
        the_file.write('stoped after epoche: '+str(len(history.history['loss']))+'\n')
        the_file.write('last loss: '+str(history.history['loss'][-1])+'\n')
        the_file.write('last detector_loss: '+str(history.history['detector_loss'][-1])+'\n')
        the_file.write('last picker_P_loss: '+str(history.history['picker_P_loss'][-1])+'\n')
        the_file.write('last picker_S_loss: '+str(history.history['picker_S_loss'][-1])+'\n')
        the_file.write('last detector_f1: '+str(history.history['detector_f1'][-1])+'\n')
        the_file.write('last picker_P_f1: '+str(history.history['picker_P_f1'][-1])+'\n')
        the_file.write('last picker_S_f1: '+str(history.history['picker_S_f1'][-1])+'\n')
        the_file.write('================== Other Parameters ========================='+'\n')
        the_file.write('label_type: '+str(ar.label_type)+'\n')
        the_file.write('augmentation: '+str(ar.augmentation)+'\n')
        the_file.write('shuffle: '+str(ar.shuffle)+'\n')               
        the_file.write('normalization_mode: '+str(ar.normalization_mode)+'\n')
        the_file.write('add_event_r: '+str(ar.add_event_r)+'\n')
        the_file.write('add_noise_r: '+str(ar.add_noise_r)+'\n')   
        the_file.write('shift_event_r: '+str(ar.shift_event_r)+'\n')                            
        the_file.write('drop_channel_r: '+str(ar.drop_channel_r)+'\n')            
        the_file.write('scale_amplitude_r: '+str(ar.scale_amplitude_r)+'\n')            
        the_file.write('pre_emphasis: '+str(ar.pre_emphasis)+'\n')