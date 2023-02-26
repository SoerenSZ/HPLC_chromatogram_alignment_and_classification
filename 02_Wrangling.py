# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:42:50 2022

@author: SEZE
"""

###############################################################################
#                                                                             #
#                            2. Data wranglin                                 #
#                                                                             #
###############################################################################


import pandas as pd


def wrangler(df_peaks, df_ph, df_resin, df_load, df_nacl, df_ion, df_conductivity, var_used = ['retention'], excel = False):
    ## 2.1 - Vial location columns
    v_row  = []
    v_col  = []
    v_flat = []
    n_rank = 20
    
    for i in df_peaks.vial_location:
        # Split
        ele = i.split(':')
        
        ## If not unknown
        if ele[0] != 'V':
            vial = ele[1].split(',')
            flat = (ord(vial[0].upper())-65) * 12 + int(vial[1])
            v_row.append(vial[0].upper())
            v_col.append(int(vial[1]))
            v_flat.append(flat)
        
        ## Unknown
        else:
            v_row.append('Nan')
            v_col.append('Nan')
            v_flat.append('Nan')
            
      
    # Add Varial to data frame
    df_peaks['vial_row'] = v_row    
    df_peaks['vial_column'] = v_col    
    df_peaks['vial_flat'] = v_flat  
    
    
    spec_id = []
    wave_length = []
    
    for i in range(len(df_peaks.channel_description)):
        ele = df_peaks.channel_description[i].split()
        
        if ele[-1][:3] == '280':
            s = float(df_peaks.vial_flat[i])+0.280
            wl = 280
    
        else:
            s = float(df_peaks.vial_flat[i])+0.214
            wl = 214
        #spec_id.append(s)
        spec_id.append(round(s,3))
        wave_length.append(wl)
        
    df_peaks['spec_id'] = spec_id
    df_peaks['wave_length'] = wave_length
    # Set vial id as str
    df_peaks['iidp_file_id'] = df_peaks['iidp_file_id'].map(str)
    print("\n2.1 - Vial row, Vial column and Vial flat inserted into DF")
    
    
    
    # add spectroscopy indentifier
    # fix vial flat in regards to nm (sinsitivity)
    
    
    if excel == True:
        ## 2.2 - pH column  
        # Possible pH values
        ph_choices = pd.unique(df_ph.pH[1:])
        pH_l = []
        
        # List of 
        for i in df_peaks.vial_location:
            # pH 5.0
            ele = i.split(':')
            if ele[0] != 'V':
                vial = ele[1].split(',')
                # pH 5.5
                if int(vial[1]) <= 4:
                    pH_l.append(ph_choices[0])
                elif int(vial[1]) > 4:
                    pH_l.append(ph_choices[1])
            else:
                pH_l.append('Nan')
        
        
        # Add pH to data frame
        df_peaks['ph'] = pH_l
        
        print("\n2.2 - pH values inserted into DF")
            
        
        
        
        ## 2.3 - Resin column  
        # Possible Resin values
        resin_l = []
        
        df_resin.Resin = df_resin.Resin[1:]
        
        for i in df_peaks.vial_column:
            if i != 'Nan':
                resin_l.append(df_resin.Resin[int(i)])
            else:
                resin_l.append('Nan')
        
        # Add Resin to data frame
        df_peaks['resin'] = resin_l    
        print("\n2.3 - Resin values inserted into DF")    
            
        
        # 2.3 Resin label
        
        resin_label = []
        uniq_resin = list(set(df_peaks['resin']))
        for i in range(len(df_peaks['resin'])):
            if df_peaks['resin'][i] in uniq_resin:
                l = uniq_resin.index(df_peaks['resin'][i])
                resin_label.append(l)
            else:
                resin_label.append(-1)
        df_peaks['resin_label'] = resin_label
        
        

        ## 2.4 - Load column  
        # Possible Load values
        load_l = []
        
        for i in df_peaks.vial_column:
            if i != 'Nan':
                load_l.append(round(df_load.Load[int(i)-1],))
            else:
                load_l.append('Nan')
        
        # Add Load to data frame
        df_peaks['load'] = load_l    
        print("\n2.4 - Load values inserted into DF")    
            
        
        
        ## 2.5 - NaCl column  
        # Possible NaCl values
        nacl_l = []
        
        for i in range(len(df_peaks.vial_column)):
            if df_peaks.vial_column[i] != 'Nan':
                c = int(df_peaks.vial_column[i])
                r = ord(df_peaks.vial_row[i])-65
                nacl = df_nacl[c][r]
                nacl_l.append(round(nacl,))
            else:
                nacl_l.append(float(0))
        
        # Add Load to data frame
        df_peaks['nacl'] = nacl_l    
        print("\n2.5 - NaCl values inserted into DF")    
            
        
        
        ## 2.6 - Ion column  
        # Possible Load values
        ion_l = []
        
        for i in df_peaks.vial_column:
            if i != 'Nan':
                ion_l.append(round(df_load.Load[int(i)-1],))
            else:
                ion_l.append('Nan')
        
        # Add Load to data frame
        df_peaks['ion'] = ion_l    
        print("\n2.6 - Ion values inserted into DF")    
            
        
        
        ## 2.7 - Conductivity column  
        # Possible Load values
        load_l = []
        
        for i in df_peaks.vial_column:
            if i != 'Nan':
                load_l.append(round(df_load.Load[int(i)-1],))
            else:
                load_l.append('Nan')
        
        # Add Load to data frame
        df_peaks['conductivity'] = load_l    
        print("\n2.7 - Load values inserted into DF")    
        
        # Remove Nan and Variables no longer needed
        df_peaks = df_peaks[df_peaks.vial_row != 'Nan']
        # Total Area
        df_t_area = df_peaks.groupby('spec_id', as_index=False)['area'].sum()
        df_t_area = df_t_area.rename(columns={'area': 'total_area'})
        df_peaks = df_peaks.merge(df_t_area, on='spec_id', how = 'left')
        
        # Test only for 280
        #for i in range(len(df_peaks.channel_description)):
         #   ele = df_peaks.channel_description[i].split()
          #  if ele[-1][:3] == '214':
           #     df_peaks = df_peaks.drop(i,axis=0)
        method = 'max'
        df_peaks[f'{method}_rank'] = df_peaks.groupby('spec_id')['area_fraction'].rank(method, ascending=False)
        df_peaks = df_peaks[df_peaks[f'{method}_rank'] <= n_rank]        
        df = df_peaks[['sample_dilution', 'area', 'asymmetry','width_50','retention','response','baseline','peak_height', 'area_fraction', 'injection_volume_unit', 'injection_volume', 'ph', 'load', 'ion', 'conductivity', 'wave_length']]
           
    else:
        # Remove Nan and Variables no longer needed
        df_peaks = df_peaks[df_peaks.vial_row != 'Nan']
        df_t_area = df_peaks.groupby('spec_id', as_index=False)['area'].sum()
        df_t_area = df_t_area.rename(columns={'area': 'total_area'})
        df_peaks = df_peaks.merge(df_t_area, on='spec_id', how = 'left')
        method = 'max'
        df_peaks[f'{method}_rank'] = df_peaks.groupby('spec_id')['area_fraction'].rank(method, ascending=False)
        df_peaks = df_peaks[df_peaks[f'{method}_rank'] <= n_rank]      
        df = df_peaks[['sample_dilution', 'area', 'asymmetry','width_50','retention','response','baseline','peak_height', 'area_fraction', 'injection_volume_unit', 'injection_volume', 'total_area', 'wave_length']]
    
    
    # Calculate <relative_area>
    df['relative_area'] = (df['area']*df['injection_volume'])/df['sample_dilution']
    df_peaks['relative_area'] = (df['area']*df['injection_volume'])/df['sample_dilution']
    #df['relative_area'] = (df['area']*df['sample_dilution'])/df['injection_volume']
    
    df = df[var_used]
    
    # Remove Nan and Variables no longer needed
    #df = df.dropna()
    df = df.reset_index()
    df_peaks = df_peaks.reset_index()
    df = df.drop(columns = ['index'])
    df_peaks = df_peaks.drop(columns = ['index'])
    print('\n|\nv')
    return df_peaks, df