
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:13:45 2022

@author: SEZE
"""

##############################################################################
#                                                                            #
#                                 0. Main                                    #
#                                                                            #
##############################################################################
import random

random.seed(1010)
"""
!!!README!!!
This program is intented to solve the issue of HPLC chromatogram peak alignment

A. Import needed packages
01. Import data
02. Data wrangling
03. Variable selection and visualisation

"""
print('\n0. Main\n____________________')


## A Libraries
try:
    lib = __import__('A_Libraries')
    #im.__all__
    print('\n0.A - Libraries loaded - SUCCELFULL')
except:
    raise Exception('\n0.A - Libraries loaded - FAILED')


## 01 Import scripts
try:
    im = __import__('01_Import')
    #im.__all__
    print('\n0.1 - Import data - SUCCELFULL')
except:
    raise Exception('\n0.1 - Import data - FAILED')



# 02 Wrangler
try:
    wr = __import__('02_Wrangling_25_01')
    #im.__all__
    print('\n0.2 - Data wrangling - SUCCELFULL')
except:
    raise Exception('\n0.2 - Data wrangling - FAILED')


# 03 Variable and model selection
try:
    ms = __import__('03_Variable_and_model_selection')
    #im.__all__
    print('\n0.3 - Model selection - SUCCELFULL')
except:
    raise Exception('\n0.3 - Model selection - FAILED')

print('\n|\nv')



   

##############################################################################
#                                                                            #
#                            A. Import Libraries                            #
#                                                                            #
##############################################################################

# Import needed libraries
os, pd, StandardScaler, sc, PCA, np, plt, KMeans, mplot3d, sn, rc, matlib, \
    itertools, linalg, mpl, mixture, DBSCAN, metrics, make_blobs, warnings = lib.Libraries()
import statistics as st
# Ignore warning
warnings.filterwarnings("ignore") 

# Standard figure resolution
plt.rcParams['figure.dpi'] = 600

##############################################################################
#                                                                            #
#                            1. Import Data                                  #
#                                                                            #
##############################################################################

print('\n1. Import Data\n____________________')



#####                           TRAIN DATA                                #####

## Import all data files (Set argument to false, if you do not have excel sheet)
file_path = r'C:/Users/SEZE/Desktop/Master_thesis_2022/Data/C0/peaks.csv'
excel_file_path = r'C:/Users/SEZE/Desktop/Master_thesis_2022/Data/Pegasus_CEXscreen_pH5.5_EXAMPLE.xlsx'

excel_train = True
## Run data importer for training data
if excel_train == True:
    df_peaks, df_ph, df_resin, df_load, df_nacl, df_ion, df_conductivity, \
        drop = im.importer(file_path, excel_file_path, excel = excel_train)
else:
    df_peaks, drop = im.importer(file_path, excel_file_path, excel = excel_train)



#####                            TEST DATA                                #####


## Import test data files (Set argument to false, if you do not have excel sheet)
file_path_test = r'C:/Users/SEZE/Desktop/Master_thesis_2022/Data/C1/peaks.csv'
excel_file_path_test = r''
print('\n1.1 - Peaks data imported - TRAINING')

excel_test = False
## Run data importer for training data
if excel_test == True:
    df_peaks_test, df_ph_test, df_resin_test, df_load_test, df_nacl_test, df_ion_test, df_conductivity_test, \
        drop_test = im.importer(file_path_test, excel_file_path_test, excel = excel_test)
else:
    df_peaks_test, drop_test = im.importer(file_path_test, excel_file_path_test, excel = excel_test)
print('\n1.2 - Peaks data imported - TEST')

print('\n|\nv')

###############################################################################
#                                                                             #
#                            2. Data wranglin                                 #
#                                                                             #
###############################################################################

print('\n2. Data wranglin\n____________________')

var_used = ['retention']

# Data wrangle - TRAINING
if excel_train == True:
    df_peaks, df = wr.wrangler(df_peaks, df_ph, df_resin, df_load, df_nacl, df_ion, df_conductivity, var_used, excel_train) 
    
else:
    df_peaks, df = wr.wrangler(df_peaks, df_ph, df_resin, df_load, df_nacl, df_ion, df_conductivity, var_used, excel_train) 


tolerance_level_list = []

for i in range(len(df_peaks['max_rank'])):
    if df_peaks.max_rank[i] == 1:
        tolerance_level_list.append(df_peaks['retention'][i])


# Add minimum size constraint
min_size = df_peaks['total_area'].max()/40

 
############### MAKE TEST FOR 214 and 280 nm seperately

w1=df_peaks['wave_length'].min()
w2=df_peaks['wave_length'].max()

# Add minimum size constraint
min_size = df_peaks['total_area'].max()/40 
min_size_1 = df_peaks[df_peaks['wave_length'] == w1]['total_area'].max()/40 
min_size_2 = df_peaks[df_peaks['wave_length'] == w2]['total_area'].max()/40 


# New try
df_peaks_old = df_peaks
df_peaks = df_peaks[(df_peaks['retention'] < 4) & (df_peaks['retention'] > 2.9)]
df_peaks_small = df_peaks_old[((df_peaks_old['total_area'] < min_size_1) & (df_peaks_old['wave_length'] == w1)) | ((df_peaks_old['total_area'] < min_size_2) & (df_peaks_old['wave_length'] == w2))]
df_peaks_ex = df_peaks_old[(~df_peaks_old['spec_id'].index.isin(df_peaks['spec_id'].index)) & (~df_peaks_old['spec_id'].index.isin(df_peaks_small['spec_id'].index))]
df_peaks_1 = df_peaks[((~df_peaks['spec_id'].index.isin(df_peaks_small['spec_id'].index)) & (~df_peaks['spec_id'].index.isin(df_peaks_ex['spec_id'].index)))]
df = df[df.index.isin(df_peaks.index)]





# Data wrangle - TEST
if excel_test == True:
    df_peaks_test, df_test = wr.wrangler(df_peaks_test, df_ph, df_resin, df_load, df_nacl, df_ion, df_conductivity, var_used, excel_train) 
    
else:
    df_peaks_test, df_test = wr.wrangler(df_peaks_test, df_ph, df_resin, df_load, df_nacl, df_ion, df_conductivity, var_used, excel_train) 




w1=df_peaks_test['wave_length'].min()
w2=df_peaks_test['wave_length'].max()

# Add minimum size constraint
min_size = df_peaks_test['total_area'].max()/40 
min_size_1 = df_peaks_test[df_peaks_test['wave_length'] == w1]['total_area'].max()/40 
min_size_2 = df_peaks_test[df_peaks_test['wave_length'] == w2]['total_area'].max()/40 



# New try
df_peaks_old_test = df_peaks_test
df_peaks_test = df_peaks_test[(df_peaks_test['retention'] < 4) & (df_peaks_test['retention'] > 2.9)]
df_peaks_small_test = df_peaks_old_test[((df_peaks_old_test['total_area'] < min_size_1) & (df_peaks_old_test['wave_length'] == w1)) | ((df_peaks_old_test['total_area'] < min_size_2) & (df_peaks_old_test['wave_length'] == w2))]
df_peaks_ex_test = df_peaks_old_test[(~df_peaks_old_test['spec_id'].index.isin(df_peaks_test['spec_id'].index)) & (~df_peaks_old_test['spec_id'].index.isin(df_peaks_small_test['spec_id'].index))]
df_peaks_test_1 = df_peaks_test[((~df_peaks_test['spec_id'].index.isin(df_peaks_small_test['spec_id'].index)) & (~df_peaks_test['spec_id'].index.isin(df_peaks_ex_test['spec_id'].index)))]
df_test = df_test[df_test.index.isin(df_peaks_test.index)]



#############################

###########################



###############################################################################
#                                                                             #
#                            3. Model selection                               #
#                                                                             #
###############################################################################

print('\n3. Model testing\n____________________')



# COMMENT OUT IF DTW IS NEEDED
#file_path = 'C:/Users/SEZE/Desktop/Master_thesis_2022/Data/C0_chromatograms/sample-set.json'
#df = ms.DTW_(df, file_path)



################### TRAIN ####################
## 3.1 - Variance explained - with a default threshold of 0.55 and for all variables
#X_train_pca, pca, X_train_std = ms.variance_explained(df, 0.99)
X_train_pca, pca, X_train_std = ms.variance_explained(df, 1)

#X_train_pca = df['Retention'].values.reshape(-1,1)

## 3.2 - K-means - default 4 classes if nonthing else is specified
classes = 2 #7
classes_p = classes
kmeans = ms.k_means(X_train_pca, classes)
#kmeans = ms.k_means(X_train_std, classes)
df['prediction'] = kmeans.labels_


## Add Prediction and peak number to df
var, df = ms.df_id_add(df, df_peaks)



## 3.5 - Label prediction, plotted on rentention time and ID
classes = len(set(kmeans.labels_))
conf = 'Null'
title = f'K-Means with: {classes} classes - Confidence:{conf}'
ms.label_predictions(df,kmeans.labels_, title)

## 3.6 - Calculate confident k-mean predictions
conf = 0.95
df_confident, res = ms.confident_k_means(df, classes, X_train_pca, kmeans, confidence = conf, tolerance = False)
title = f'K-Means with: {classes} classes - Confidence:{int(conf*100)}%'
ms.label_predictions(df_confident,kmeans.labels_, title)

#df_confident, res = ms.confident_k_means(df, classes, X_train_std, kmeans, confidence = conf, tolerance = False)

# Excluded due to confidence level or clustered as the same as a higher confidenced peak in the same class at the same idx
ex_conf = df_peaks[(~df_peaks['spec_id'].index.isin(res['spec_id'].index))]

# Outliers, out of interest, low confidence
ex_peaks_l = [df_peaks_small, df_peaks_ex, ex_conf]
ex_peaks_labels = ['Peaks Size', 'Outlier', 'Confidence']
ex_peaks_count = [len(df_peaks_small), len(df_peaks_ex), len(ex_conf)]




# Plot vial, retention time and classification
# Select the two variables to be plotted
res = res.sort_values(['prediction', 'spec_id'])
var1 = 'spec_id'
var2 =  'retention'
w1=df_peaks['wave_length'].min()
w2=df_peaks['wave_length'].max()
x1 = res.loc[df_peaks['wave_length'] == w1][var1]
y1 = res.loc[df_peaks['wave_length'] == w1][var2]
x2 = res.loc[df_peaks['wave_length'] == w2][var1]
y2 = res.loc[df_peaks['wave_length'] == w2][var2]
l1 = res.loc[df_peaks['wave_length'] == w1]['prediction']
l2 = res.loc[df_peaks['wave_length'] == w2]['prediction']
t1 = f'K-means with {classes} labels - {w1}nm'
t2 = f'K-means with {classes} labels - {w2}nm'
x_label = 'ID'
y_label = 'Rt'
upper_bound = 4
lower_bound = 2.9



# Excluded

size_1 = df_peaks_small.loc[df_peaks_small['wave_length'] == w1][[var1,var2]]
outliers_1 = df_peaks_ex.loc[df_peaks_ex['wave_length'] == w1][[var1,var2]]
confidence_1 = ex_conf.loc[ex_conf['wave_length'] == w1][[var1,var2]]
ex_peaks_l_1 = [size_1, outliers_1, confidence_1]
ex_peaks_count_1 = [len(size_1), len(outliers_1), len(confidence_1)]

size_2 = df_peaks_small.loc[df_peaks_small['wave_length'] == w2][[var1,var2]]
outliers_2 = df_peaks_ex.loc[df_peaks_ex['wave_length'] == w2][[var1,var2]]
confidence_2 = ex_conf.loc[ex_conf['wave_length'] == w2][[var1,var2]]
ex_peaks_l_2 = [size_2, outliers_2, confidence_2]
ex_peaks_count_2 = [len(size_2), len(outliers_2), len(confidence_2)]

ex_labels = ['Outlier', 'Confidence band', 'Dublicated labels']


ms.plotter_to_use(x1, y1, x2, y2, classes, l1, l2, upper_bound, lower_bound, ex_labels, ex_peaks_l_1, ex_peaks_count_1, ex_peaks_l_2, ex_peaks_count_2,  t1, t2, x_label, y_label)





################### TEST ####################
## 3.1 - Variance explained 
X_train_pca, pca, X_train_std = ms.variance_explained(df_test, 1)


## 3.2 - K-means - default 4 classes if nonthing else is specified
classes = 2 
classes_p = classes
kmeans = ms.k_means(X_train_pca, classes)
df_test['prediction'] = kmeans.labels_


## Add Prediction and peak number to df
var, df = ms.df_id_add(df_test, df_peaks_test)



## 3.5 - Label prediction, plotted on rentention time and ID
classes = len(set(kmeans.labels_))
conf = 'Null'
title = f'K-Means with: {classes} classes - Confidence:{conf}'
ms.label_predictions(df,kmeans.labels_, title)

## 3.6 - Calculate confident k-mean predictions
conf = 0.95
df_confident, res = ms.confident_k_means(df, classes, X_train_pca, kmeans, confidence = conf, tolerance = False)


# Excluded due to confidence level or clustered as the same as a higher confidenced peak in the same class at the same idx
ex_conf_test = df_peaks_test[(~df_peaks_test['spec_id'].index.isin(res['spec_id'].index))]

# Outliers, out of interest, low confidence
ex_peaks_l = [df_peaks_small_test, df_peaks_ex_test, ex_conf_test]
ex_peaks_labels = ['Peaks Size', 'Outlier', 'Confidence']
ex_peaks_count = [len(df_peaks_small_test), len(df_peaks_ex_test), len(ex_conf_test)]



# Plot vial, retention time and classification
# Select the two variables to be plotted
res = res.sort_values(['prediction', 'spec_id'])
var1 = 'spec_id'
var2 =  'retention'
w1=df_peaks_test['wave_length'].min()
w2=df_peaks_test['wave_length'].max()
x1 = res.loc[df_peaks_test['wave_length'] == w1][var1]
y1 = res.loc[df_peaks_test['wave_length'] == w1][var2]
x2 = res.loc[df_peaks_test['wave_length'] == w2][var1]
y2 = res.loc[df_peaks_test['wave_length'] == w2][var2]
l1 = res.loc[df_peaks_test['wave_length'] == w1]['prediction']
l2 = res.loc[df_peaks_test['wave_length'] == w2]['prediction']
t1 = f'K-means with {classes} labels - {w1}nm'
t2 = f'K-means with {classes} labels - {w2}nm'
x_label = 'ID'
y_label = 'Rt'
upper_bound = 4
lower_bound = 2.9

# Excluded
size_1 = df_peaks_small_test.loc[df_peaks_small_test['wave_length'] == w1][[var1,var2]]
outliers_1 = df_peaks_ex_test.loc[df_peaks_ex_test['wave_length'] == w1][[var1,var2]]
confidence_1 = ex_conf_test.loc[ex_conf_test['wave_length'] == w1][[var1,var2]]
ex_peaks_l_1 = [size_1, outliers_1, confidence_1]
ex_peaks_count_1 = [len(size_1), len(outliers_1), len(confidence_1)]

size_2 = df_peaks_small_test.loc[df_peaks_small_test['wave_length'] == w2][[var1,var2]]
outliers_2 = df_peaks_ex_test.loc[df_peaks_ex_test['wave_length'] == w2][[var1,var2]]
confidence_2 = ex_conf_test.loc[ex_conf_test['wave_length'] == w2][[var1,var2]]
ex_peaks_l_2 = [size_2, outliers_2, confidence_2]
ex_peaks_count_2 = [len(size_2), len(outliers_2), len(confidence_2)]

ex_labels = ['Outlier', 'Confidence band', 'Dublicated labels']
ms.plotter_to_use(x1, y1, x2, y2, classes, l1, l2, upper_bound, lower_bound, ex_labels, ex_peaks_l_1, ex_peaks_count_1, ex_peaks_l_2, ex_peaks_count_2,  t1, t2, x_label, y_label)



###############################################################################
#                                                                             #
#                            4. DRIFT WITH DTW                                #
#                                                                             #
###############################################################################


################################ TRAIN ########################################

df_drift = res.sort_values(by=['spec_id'])
df_drift = res.sort_values(['prediction', 'spec_id'])

# Unique ID's
ids = np.sort(list(set(df_peaks['spec_id'])))

step = 0.02500000037252903

drift_rt = []

# Add drift to C0 for simmulation purposes
for i in range(len(ids)):
    multiplier = int(str(ids[i]).split('.')[0])
    for j in range(len(df_drift[df_drift['spec_id'] == ids[i]])):
        #df_drift[df_drift['spec_id'] == ids[i]].iloc[j]['retention_drift'] = df_drift[df_drift['spec_id'] == ids[i]].iloc[j]['retention']*step*multiplier
        drift_rt.append(df_drift[df_drift['spec_id'] == ids[i]].iloc[j]['retention']+step*multiplier)
    
    
df_drift = df_drift.assign(rt_drift = drift_rt)    

    

# Excluded due to confidence level or clustered as the same as a higher confidenced peak in the same class at the same idx
ex_conf = df_peaks[(~df_peaks['spec_id'].index.isin(res['spec_id'].index))]

# Outliers, out of interest, low confidence
ex_peaks_l = [df_peaks_small, df_peaks_ex, ex_conf]
ex_peaks_labels = ['Peaks Size', 'Outlier', 'Confidence']
ex_peaks_count = [len(df_peaks_small), len(df_peaks_ex), len(ex_conf)]




# Plot vial, retention time and classification
# Select the two variables to be plotted
res = res.sort_values(['prediction', 'spec_id'])
var1 = 'spec_id'
var2 =  'rt_drift'
w1=df_peaks['wave_length'].min()
w2=df_peaks['wave_length'].max()
x1 = df_drift.loc[df_peaks['wave_length'] == w1][var1]
y1 = df_drift.loc[df_peaks['wave_length'] == w1][var2]
x2 = df_drift.loc[df_peaks['wave_length'] == w2][var1]
y2 = df_drift.loc[df_peaks['wave_length'] == w2][var2]
l1 = df_drift.loc[df_peaks['wave_length'] == w1]['prediction']
l2 = df_drift.loc[df_peaks['wave_length'] == w2]['prediction']
t1 = f'K-means with {classes} labels - {w1}nm'
t2 = f'K-means with {classes} labels - {w2}nm'
x_label = 'ID'
y_label = 'Rt'
upper_bound = 4
lower_bound = 2.9

var2 = 'retention'

# Excluded
size_1 = df_peaks_small.loc[df_peaks_small['wave_length'] == w1][[var1,var2]]
outliers_1 = df_peaks_ex.loc[df_peaks_ex['wave_length'] == w1][[var1,var2]]
confidence_1 = ex_conf.loc[ex_conf['wave_length'] == w1][[var1,var2]]
ex_peaks_l_1 = [size_1, outliers_1, confidence_1]
ex_peaks_count_1 = [len(size_1), len(outliers_1), len(confidence_1)]

size_2 = df_peaks_small.loc[df_peaks_small['wave_length'] == w2][[var1,var2]]
outliers_2 = df_peaks_ex.loc[df_peaks_ex['wave_length'] == w2][[var1,var2]]
confidence_2 = ex_conf.loc[ex_conf['wave_length'] == w2][[var1,var2]]
ex_peaks_l_2 = [size_2, outliers_2, confidence_2]
ex_peaks_count_2 = [len(size_2), len(outliers_2), len(confidence_2)]

ex_labels = ['Outlier', 'Confidence band', 'Dublicated labels']
ms.plotter_to_use(x1, y1, x2, y2, classes, l1, l2, upper_bound, lower_bound, ex_labels, ex_peaks_l_1, ex_peaks_count_1, ex_peaks_l_2, ex_peaks_count_2,  t1, t2, x_label, y_label)











