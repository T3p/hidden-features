#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 14:17:33 2019

@author: Matteo Papini
"""
import tensorboardX as tbx
import csv
import warnings
import os


class Logger():
    def __init__(self, directory='./logs_t', name='auto', modes=['human', 'csv', 'tensorboard']):
        self.modes = modes
        self.directory = directory
        self.name = name
        self.ready = False
        self.keys = []
        self.open_files = []
        
    def open(self, keys):
        try:
            maybe_make_dir(self.directory)
            self.keys = keys
        except:
            warnings.warn('Could not create log directory!')
            return
            
        # csv
        if 'csv' in self.modes:
            try:
                self.csv_file = open(self.directory + '/' 
                                     + self.name + '.csv', 'w')
                self.open_files.append(self.csv_file)
                self.csv_writer = csv.DictWriter(self.csv_file, keys)
                self.csv_writer.writeheader()
                self.ready = True
            except:
                warnings.warn('Could not create csv file!')
        
            if 'tensorboard' in self.modes:
                try:
                    self.tb_writer = tbx.SummaryWriter(self.directory + '/'
                                                       + self.name)
                    self.ready = True
                except:
                    warnings.warn('Could not create TensorboardX files!')
            
            if 'human' in self.modes:
                self.ready = True
        
    def write_row(self, row, iteration):
        if not self.ready:
            warnings.warn('You must open the logger first!')
            return
        
        if 'human' in self.modes:
            print('\nIteration %d' % iteration)
            for key, val in row.items():
                print(key, ':\t', val)
        
        #csv
        if 'csv' in self.modes:
            try:
                self.csv_writer.writerow(row)
            except:
                warnings.warn('Could not write data to csv!')
                
        if 'tensorboard' in self.modes:
            try:
                for key, val in row.items():   
                    self.tb_writer.add_scalar(key, val, iteration)
            except:
                warnings.warn('Could not write data to TensorboardX')
        
    def close(self):
        if not self.ready:
            warnings.warn('You must open the logger first!')
            return
        
        try:
            for f in self.open_files:
                f.close()
            self.ready = False
        except:
            warnings.warn('Could not close logger!')

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)