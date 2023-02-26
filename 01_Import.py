# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 09:35:23 2022

@author: SEZE
"""




##############################################################################
#                                                                            #
#                            1. Import Data                                  #
#                                                                            #
##############################################################################

# Libraries
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Importer function
def importer(file_path, excel_file_path, excel = True):
    

    ## 1.1 - Peaks
    try:
        if os.stat(file_path).st_size > 0:
            df_peaks = pd.read_csv(file_path)
        else:
            print('Empty file.....')
    except OSError:
        print('No such file excist in this directory')
    if excel == True:
        try:
            if os.stat(excel_file_path).st_size > 0:
                ## 1.2 - pH 
                df_ph = pd.read_excel (excel_file_path, sheet_name='pH')
                print('\n1.2 - pH data imported')
                
                ## 1.3 - Resin 
                df_resin = pd.read_excel (excel_file_path, sheet_name='Resin')
                print('\n1.3 - Resin data imported')
                
                ## 1.4 - Load 
                df_load = pd.read_excel (excel_file_path, sheet_name='Load')
                print('\n1.4 - Load data imported')
                
                ## 1.5 - NaCl 
                df_nacl = pd.read_excel (excel_file_path, sheet_name='NaCl')
                print('\n1.5 - NaCL data imported')
                
                
                ## 1.6 - Ion_concentration 
                df_ion = pd.read_excel (excel_file_path, sheet_name='Ion_concentration')
                print('\n1.6 - Ion concentration data imported')
                
                
                ## 1.7 - Conductivity
                df_conductivity = pd.read_excel (excel_file_path, sheet_name='Conductivity')
                print('\n1.7 - Conductivity data imported')
                
                drop = ['response', 'injection_volume_unit',  'baseline', 'conductivity', 'width_50','peak_height', 'asymmetry','ion', 'load', 'ph']
            else:
                print('Empty file.....')
        except OSError:
            print('No such file excist in this directory')
        return df_peaks, df_ph, df_resin, df_load, df_nacl, df_ion, df_conductivity, drop
        
        
    else:
        #drop = ['response','injection_volume', 'injection_volume_unit', 'baseline', 'width_50','peak_height', 'asymmetry']
        drop = ['response', 'injection_volume_unit', 'baseline', 'width_50','peak_height', 'asymmetry']
        
        return df_peaks, drop