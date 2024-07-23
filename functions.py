#@author M. Schultheiß, T. Dorosti, P. Schmette, J. Heuchert
import numpy as np
import os
import pandas as pd
import math
import h5py
import matplotlib.pyplot as plt
import random
import astra
import xraylib as xr
import spekpy as sp
from lungmask import mask # Note: lungmask can use GPU for faster calculations (via torch)
import torch
from skimage.transform import rescale, resize
from configuration_paper import CURRENT_GPU, POSITION, CURRENT_CONFIG, PATH_SERVER_CACHE
from spectrum_tools import get_attenuation
from lib_polychromsimulation import *
import tensorflow as tf
import datetime
from scipy import ndimage
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Conv2D
from keras.preprocessing.image import ImageDataGenerator
import keras.initializers
from keras.models import load_model
from sliceClasses import Slice2D, Slice2DSet

def segment_body_from_vol(vol,threshold=-775):
    '''
    Removes the patient table in the scan. 
    @author P. Schmette
    '''
    import scipy.ndimage as ndimage
    for i in range(vol.shape[0]):
        img = vol[i]
        mask = (img>threshold).astype(int)
        mask = ndimage.binary_opening(mask,iterations=2)
        label, num_label = ndimage.label(mask)
        size = np.bincount(label.ravel())
        biggest_label = size[1:].argmax() + 1
        mask = label == biggest_label
        mask = ndimage.binary_fill_holes(mask)
        vol[i] = (mask*img)+(-1024)*(1-mask) # 
    return vol.clip(-1024,  None)

def getScaleFactor(vol_shape, voxel_size_z, voxel_size_x): #put in a general functions script
    '''
    vol_shape (triple tuple): CT volume shape to get the scaling factor for
    voxel_size_z (float): pixel spacing along axis 0 in above diagram
    voxel_size_x (float): pixel spacing along axis 1 in above diagram 
    output: scale factor (float) for correcting shape of CT into an isotropic volume
    '''
    num_slices_z, num_slices_x = vol_shape[0], vol_shape[1]
    z_total_mm = voxel_size_z * num_slices_z
    x_total_mm = voxel_size_x * num_slices_x
    return x_total_mm/z_total_mm

def generateSims(datapath, save_dir, name, position, config_name, pathCTs):
    '''
    datapath (str): path to csv file with matadata and individual CT file names
    save_dir (str): general directory to save simulated data to
    name (str): name of the public dataset (to be used for the final save directory)
    position (str): keys 'pa' (frontal) or 'lat' (lateral) for position of radiographs
    config_name (str): name of the current configuration
    pathCTs (str): path to directory where CT data is saved
    output: doesn't return anything, but saved simulated projections to the given directory
    '''
    # get CT data and metadata info, define paths:
    df = pd.read_csv(datapath)
    save_dir += name+'/lungthickness/'
    path_lung_projections_polychromatic = save_dir+position+"_"+name+"_projections_polychromatic"
    
    # making a new folder for configuration is necessary:
    if not os.path.exists(path_lung_projections_polychromatic+"_"+config_name):
        os.mkdir(path_lung_projections_polychromatic+"_"+config_name)

    # iterate through all CT volumes:
    for index, row in df.iterrows():
        if row['KVP']!=120 : # only use 120 kvp scans 
            print('{} has KVP = {} ==> skipped '.format(row['identifier'], row['KVP']))
            continue
        
        # get CT:
        ct = np.load(os.path.join(pathCTs, str(row['vol_file'])), mmap_mode='c')
        ct = segment_body_from_vol(ct) #get rid of patient table in the volume
        # voxel size correction so that the size matches in all directions:
        scale_factor = getScaleFactor(ct.shape, row["spacing_0"], row["spacing_1"])
        ct = resize(serie,(np.round(ct.shape[0]*scale_factor), ct.shape[1], ct.shape[2]), preserve_range=True)
        # get lung segmentation:
        segmentation = mask.apply(ct) # generate segmentation
        segmentation[segmentation>1]=1 # making both lung lobes have value 1 (combine the 2)
        voxel_length = row["spacing_1"]  
        
        if position=='lat': #lateral images
            ct = np.swapaxes(ct, 0,2)
            ct = np.swapaxes(ct, 0,1)
            segmentation = np.swapaxes(segmentation, 0,2)
            segmentation = np.swapaxes(segmentation, 0,1)
            voxel_length = row["spacing_2"]

        # get projections:
        ct_proj = get_projections_polychromatic(ct, voxel_length, kvp=cparam("KVP"))
        lung_proj = get_projections_new(segmentation, voxel_size_isotrophic=voxel_length)  * voxel_length
        # save simulations as hdf5:
        outpath = os.path.join(path_lung_projections_polychromatic+"_"+config_name, row['identifier']+".hdf5")
        print("Synthetic data shape is", lung_proj.shape)
        save_hdf5_projections(outpath, ct_proj, lung_proj)
        print("done with ", outpath)

def save_dataSplitCsv(save_dir, name, position, config_name):
    '''
    save_dir, (str): general directory to save simulated data to
    name (str): name of the public dataset (to be used for the final save directory)
    position (str): keys 'pa' (frontal) or 'lat' (lateral) for position of radiographs
    config_name (str): name of the current configuration
    '''
    path_lung_projections = save_dir+position+"_"+name+"_projections_polychromatic_"+config_name
        
    l = os.listdir(path_lung_projections)
    l = [i for i in l if not i.endswith('.csv')] # take care to remove the .csv metadata file, if saved under the same directory
    random.Random(4).shuffle(l) #random shuffle nums with seed=4
    
    df_train = pd.DataFrame(l[:int(0.6*len(l))]) # ~60% 
    df_val = pd.DataFrame(l[int(0.6*len(l)):int(0.8*len(l))]) # ~20% 
    df_test = pd.DataFrame(l[int(0.8*len(l)):]) # ~20% 
    
    df_train.to_csv('./splits/lungvolume{}_train.csv'.format(name), header=False, index=False)
    df_val.to_csv('./splits/lungvolume{}_val.csv'.format(name), header=False, index=False)
    df_test.to_csv('./splits/lungvolume{}_test.csv'.format(name), header=False, index=False)

def getSyntheticData(datapath, save_dir, position, name, config_name, given_range, return_pixelSpacing=False): 
    '''
    datapath (str): path to csv file with matadata and individual file names
    save_dir (str): general directory where simulated data is saved
    name (str): name of the public dataset (as used for the final save directory)
    position (str): keys 'pa' (frontal) or 'lat' (lateral) for position of radiographs
    config_name (str): name of the current configuration
    given_range (range): the range of projections to retrieve for each radiograph
    
    output: 
    slices (Slice2D class) holding the radiograph and thickness slices 
    optionally pixel spacing in 
    '''
    df = pd.read_csv(datapath)
    path_lung_projections = save_dir+position+"_"+name+"_projections_polychromatic_"+config_name
    slices = Slice2DSet()

    # iterate through all data:
    for index, row in df.iterrows():
        if row['KVP']!=120: # only use 120 kvp scans 
            continue
            
        projpath = os.path.join(path_lung_projections, row['identifier']+".hdf5")
        if not os.path.exists(projpath):
                print("Warning: Path not found:", projpath)
        else:
            hf = h5py.File(projpath, 'r')
            print("Load: ", projpath)
            for idx in given_range: #loat based on the given range (all 10 radiographs or only central ones)
                newshape = [cparam("IMAGEWIDTH"),cparam("IMAGEWIDTH")]
                data_lungmask = hf.get('lung_'+str(idx))[()] 
                data_radiograph = hf.get('radiograph_'+str(idx))[()]
                data_radiograph = 8*(data_radiograph/data_radiograph.max())#normalize radiographs to 0-8:
                
                # voxel size correction so that the size matches in all directions:
                scale_factor = getScaleFactor((row["num_slices"], data_radiograph.shape[0], data_radiograph.shape[1]), row["spacing_0"], row["spacing_1"])
                data_radiograph_scaled = rescale(data_radiograph, (1/scale_factor, scale_factor), preserve_range=True)
                data_lungmask_scaled = rescale(data_lungmask, (1/scale_factor, scale_factor), preserve_range=True)
            
                if scale_factor >1.0: # make sure to go back to the original shape after stretching
                    radiograph_scaled_sh = data_radiograph_scaled.shape
                    lungmask_scaled_sh = data_lungmask_scaled.shape
                    data_radiograph_scaled = data_radiograph_scaled[:,radiograph_scaled_sh[0]//2-data_radiograph.shape[0]//2:radiograph_scaled_sh[1]//2+data_radiograph.shape[1]//2]
                    data_lungmask_scaled = data_lungmask_scaled[:,lungmask_scaled_sh[0]//2-data_radiograph.shape[0]//2:lungmask_scaled_sh[1]//2+data_radiograph.shape[1]//2]

                data_radiograph_scaled = resize(data_radiograph, newshape, preserve_range=True)
                data_lungmask_scaled = resize(data_lungmask, newshape, preserve_range=True)   
                sl = Slice2D(data_radiograph_scaled, class_label = data_lungmask_scaled , filename=str((row['identifier'])))
                slices.append(sl)
            hf.close()
    return slices

def mse(gt, p, std=False):
    '''
    Mean squared error between ground truth and prediction
    '''
    if std:
        return ((np.array(gt) - np.array(p))**2).mean(), ((np.array(gt) - np.array(p))**2).std()
    else:
        return ((np.array(gt) - np.array(p))**2).mean()
        
def mae(gt, p, std=False):
    '''
    Mean absolute error between ground truth and prediction
    '''
    mae_difference = np.abs(np.array(gt) - np.array(p))
    if std:
        return np.mean(mae_difference), np.std(mae_difference)
    else:
        return np.mean(mae_difference)

def mape(gt, p):
    '''
    Mean absolute percentage error between ground truth and prediction
    '''
    return 100*np.mean(np.abs(np.array(gt) - np.array(p))/len(gt))

def get_metric(l, metric):
    import scipy.stats as st
    if metric=='mean':
        return (np.mean(np.array(l)))
    elif metric=='95ci': # 95% confidence interval
         return st.t.interval(0.95, len(l)-1, loc=np.mean(l), scale=st.sem(l))
    elif metric=='minMaxRange':
        return np.min(l), np.max(l)
    elif metric=='std': # standard deviation
        return np.std(np.array(l))

# Normality Tests: from https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
# Shapiro-Wilk Test 
def shapiro_wilk(data):
    from numpy.random import seed
    from numpy.random import randn
    from scipy.stats import shapiro
    # seed the random number generator
    seed(1)
    # generate univariate observations
    data = 5 * randn(100) + 50
    # normality test
    stat, p = shapiro(data)
    #print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Sample looks Gaussian') #(fail to reject H0)
    else:
    	print('Sample does not look Gaussian') # (reject H0)
        
def dAgostino_pearson(data):
    # D'Agostino and Pearson's Test
    from numpy.random import seed
    from numpy.random import randn
    from scipy.stats import normaltest
    # seed the random number generator
    seed(1)
    # generate univariate observations
    data = 5 * randn(100) + 50
    # normality test
    stat, p = normaltest(data)
    #print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian') # (fail to reject H0)'
    else:
        print('Sample does not look Gaussian') #(reject H0)

def anderson_darling(data):
# Anderson-Darling Test
    from numpy.random import seed
    from numpy.random import randn
    from scipy.stats import anderson
    # seed the random number generator
    seed(1)
    # generate univariate observations
    data = 5 * randn(100) + 50
    # normality test
    result = anderson(data)
    print('Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
     sl, cv = result.significance_level[i], result.critical_values[i]
     if result.statistic < result.critical_values[i]:
        #print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        print('data looks normal')
     else:
        #print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        print('data does not look normal')

def plotter(testx, groundtruth, results_predicted, y_idx = np.linspace(0,11,11), save=False, name=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.gridspec as gridspec
    def plot3(theindex):
        pass
    plt.clf()
    fig2 = plt.figure(figsize=[17,7])
    rows, cols = 4,13
    max_prediction = 0
    max_gt = 0
    max_diff = 0
    
    for y in range(0,10):
        max_prediction = max(max_prediction, results_predicted[int(y_idx[y])].max())
        max_gt = max(max_gt, groundtruth[int(y_idx[y])].max())
        max_diff = max(max_diff, np.abs(groundtruth[int(y_idx[y])]-results_predicted[int(y_idx[y])]).max())

    def addcb(im3, cax, text=""):
        if False:
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            cb = plt.colorbar(im3, cax=cax, orientation='vertical');
            if text!="":
                cb.set_label(text)
        else:
            cb = plt.colorbar(im3);
            if text!="":
                cb.set_label(text)

    grid = gridspec.GridSpec(ncols=11, nrows=4, figure=fig2, width_ratios=[0.1,1, 1,1,1,1,1,1,1,1,1.25])

    names = ["A", "B", "C", "D"] 
        ax1 = plt.subplot(grid[i,0])  
        ax1.set_axis_off()
        ax1.text(0, 0.5,  names[i]);
    from matplotlib.colors import ListedColormap

    softstart = 40
    cmap = plt.cm.viridis
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[0:softstart,3]    = np.linspace(0,1, softstart)
    my_cmap = ListedColormap(my_cmap)

    cmap2 = plt.cm.inferno
    my_cmap2 = cmap2(np.arange(cmap2.N))
    my_cmap2[0:1,3]    = np.linspace(0,1, 1)
    my_cmap2 = ListedColormap(my_cmap2)

    for y in range(1,11):
        ax1 = plt.subplot(grid[0, y])                 
        ax1.set_anchor('W')
        
        im3 = plt.imshow(testx[int(y_idx[y])].transpose(1,0), cmap="gray", aspect="equal")
        plt.axis('off')

        ax1.xaxis.set_label_position('top')
        ax1.set_title("Case "+str(y))   
        ax1 = plt.subplot(grid[1, y])                 
        ax1.set_anchor('W')

        im3 = plt.imshow(results_predicted[int(y_idx[y])].transpose(1,0),
                         vmin=0,vmax=max_prediction,  cmap=my_cmap, aspect="equal")
        plt.axis('off')
        if y == 10:
            addcb(im3, ax1, text="Thickness [mm]")

        ax1 = plt.subplot(grid[2, y])                 
        im3 = plt.imshow(testx[int(y_idx[y])].transpose(1,0), cmap="gray", aspect="equal")
        im3 = plt.imshow(groundtruth[int(y_idx[y])].transpose(1,0),
                         vmin=0,vmax=max_prediction,  cmap=my_cmap, aspect="equal")
        plt.axis('off')
        if y == 10:
            addcb(im3, ax1, text="Thickness [mm]")
        ax1.set_anchor('W')
        ax1 = plt.subplot(grid[3, y])                 
        im3 = plt.imshow(np.abs(groundtruth[int(y_idx[y])]-results_predicted[int(y_idx[y])]).transpose(1,0),
                         vmin=0,vmax=max_diff,  cmap="inferno", aspect="equal")
        plt.axis('off')
        ax1.set_anchor('W')
        if y == 10:
            addcb(im3, ax1, text="Thickness Difference [mm]")

    plt.tight_layout(pad=1.0, w_pad=0.1, h_pad=0.1)
    if save:
        plt.savefig("{}.pdf".format(name), format='pdf', transparent=False)

# processes (stretch and clip) PE data so that it looks right when plotted
def process_PE(i, img, pixel_spacings, slices_z):
    sh = rescale(img[i], pixel_spacings[i][2]/pixel_spacings[i][1]).shape[0] //2
    return resize(rescale(img[i], pixel_spacings[i][2]/pixel_spacings[i][1])[:,sh-slices_z[i]//2:sh+slices_z[i]//2], (256, 256))