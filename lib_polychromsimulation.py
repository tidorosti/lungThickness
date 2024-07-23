# -*- coding: utf-8 -*-
# @author M. Schultheiss

import numpy as np
import skimage.transform
import os
import pandas as pd
from os import listdir
from os.path import isfile, join
from skimage.util import crop
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
import os
import astra
import numpy as np

import math
import h5py
import os
import xraylib as xr
import math, astra
from spectrum_tools import get_attenuation
import spekpy as sp
import matplotlib.pyplot as plt
from configuration_paper import cparam


# To use astra, see example at:
# https://github.com/astra-toolbox/astra-toolbox/blob/10d87f45bc9311c0408e4cacdec587eff8bc37f8/samples/python/s020_3d_multiGPU.py


def get_volume_from_projection2(projection_image):
    """
        Get volume from projection given a certain geometry
    """
    assert(projection_image.shape[0] == 512)
    OD_dist = cparam("PATIENT_TO_DETECTOR")
    SO_dist = cparam("SOURCE_TO_DETECTOR") - cparam("PATIENT_TO_DETECTOR")

    SD_Dist = SO_dist+OD_dist
    Magnification = SO_dist/SD_Dist # < 1
    
    pixel_size_x = float(cparam("DETECTOR_WIDTH")/512)
    pixel_size_y = float(cparam("DETECTOR_HEIGHT")/512)

    return (((projection_image.sum()) * (pixel_size_x  * Magnification ) * (pixel_size_y * Magnification    )))  # do not mulltyply by mm3 but by mm2 in the nd as thickness is already one dimeinson

    
def get_volume_from_projection(projection_image, voxel_size_isotrophic):
    OD_dist = cparam("PATIENT_TO_DETECTOR")/voxel_size_isotrophic
    SO_dist = cparam("SOURCE_TO_DETECTOR")/voxel_size_isotrophic - cparam("PATIENT_TO_DETECTOR")/voxel_size_isotrophic

    SD_Dist = SO_dist+OD_dist
    Magnification = SO_dist / SD_Dist # < 1
    
    pixel_size_x = float(cparam("DETECTOR_WIDTH")/512/voxel_size_isotrophic)
    pixel_size_y = float(cparam("DETECTOR_HEIGHT")/512/voxel_size_isotrophic)

    return (((projection_image.sum()) * (pixel_size_x  * Magnification ) * (pixel_size_y * Magnification    )) * (voxel_size_isotrophic * voxel_size_isotrophic ))  # do not mulltyply by mm3 but by mm2 in the nd as thickness is already one dimeinson


def get_projections_new(volume, voxel_size_isotrophic,  gpu=2):
    OD_dist = cparam("PATIENT_TO_DETECTOR")/voxel_size_isotrophic
    SO_dist = cparam("SOURCE_TO_DETECTOR")/voxel_size_isotrophic - cparam("PATIENT_TO_DETECTOR")/voxel_size_isotrophic

    SD_Dist = SO_dist+OD_dist
    Magnification = SO_dist/SD_Dist # < 1

    vol_geom = astra.create_vol_geom(volume.shape[1], volume.shape[2], volume.shape[0])

    # augment projections from 10 angles
    angles_degree =  np.linspace(-10, 10, 10)
    angles = list(map(math.radians, angles_degree))
    
    # Divide by voxel_size_isotrophic in order to normalize to projection geometry
    # (e.g. voxels have size 1). See test_compare_ct_to_lv for a more detailed explanation.
    proj_geom = astra.create_proj_geom("cone",float(cparam("DETECTOR_WIDTH")/512/voxel_size_isotrophic), float(cparam("DETECTOR_HEIGHT")/512/voxel_size_isotrophic), 512,512, angles, SO_dist, OD_dist) 
    proj_id, proj_data = astra.create_sino3d_gpu(volume, proj_geom, vol_geom,  returnData=True, gpuIndex=gpu)
    projections = proj_data.transpose(1,2,0)
    astra.data3d.delete(proj_id)
    return projections


def get_rho_str(hu_ct, mu_per_rho_tissue):
   """
      Estimate density from CT units.
      mu/rho of tissue is for CT scanner spectrum
   """
   MU_WATER_KEV =cparam( "ATT_WATER") # at around 60 kev mu/rho = 2.059E-01 with rho_water = 1
   rho_str = (hu_ct / 1000 * MU_WATER_KEV + MU_WATER_KEV)  / mu_per_rho_tissue
   return rho_str


def get_rho_map(kvp, tissue, ct_volume_in_hu, mask,  voxel_size_isotrophic=1,gpu_index=2):
   """
      Return spectrum and volume with densities for a given tissue type.

      Args:
      - kVp: peak voltage of simulated spectrum
      - tissue: NIST tissue descriptor, e.g. "Bone, Cortical (ICRP)"
      - ct_volume_in_hu: CT volume mask: index 0 must be z axis, index 1 and 2 must be x and y axis. 
      - mask: a binary mask of the selected tissue (i.e., bones)

   """

   s2=sp.Spek(kvp=kvp,th=24)

   # Simulate aluminum filter and air
   s2.filter('Al',cparam("DETECTOR_AL")).filter('Air',1000)
  
   kvp_array_original, spectrum_bincount = s2.get_spectrum(edges=False)
   kvp_array = kvp_array_original.copy()

   # Cesium Iodid Scintilator
   qe_csi = 1 - np.exp(-get_attenuation(["CsI"], kvp_array)[0] * cparam("DETECTOR_CSI_DENSITY") * cparam("DETECTOR_CSI_TICKNESS") * 0.1)
   
   # 0.1 to convert between mm and cm, i.e., 6mm = 0.6 cm
   final_spectrum_unnormalized = qe_csi * spectrum_bincount
   final_spectrum = final_spectrum_unnormalized/final_spectrum_unnormalized.sum()

   att_tissue= get_attenuation([tissue], kvp_array)[0]  # returns mu/rho for whole spectrum

   if tissue=="Bone, Cortical (ICRP)":
      density_vol = mask * get_rho_str(ct_volume_in_hu, cparam("ATT_BONE") )
      return density_vol, kvp_array, final_spectrum, att_tissue, get_projections_new(density_vol, voxel_size_isotrophic=voxel_size_isotrophic, gpu=gpu_index), get_projections_new(mask,voxel_size_isotrophic=voxel_size_isotrophic, gpu=gpu_index) 

   elif tissue=="Tissue, Soft (ICRP)":
      density_vol =  mask *  get_rho_str(ct_volume_in_hu, cparam("ATT_SOFTTISSUE"))
      return density_vol, kvp_array, final_spectrum, att_tissue,  get_projections_new(density_vol, voxel_size_isotrophic=voxel_size_isotrophic, gpu=gpu_index), get_projections_new(mask,voxel_size_isotrophic=voxel_size_isotrophic, gpu=gpu_index) 

   elif tissue=="Adipose Tissue (ICRP)":
      density_vol =  mask *  get_rho_str(ct_volume_in_hu, cparam("ATT_ADIPOSE"))
      return density_vol, kvp_array, final_spectrum, att_tissue,  get_projections_new(density_vol, voxel_size_isotrophic=voxel_size_isotrophic, gpu=gpu_index), get_projections_new(mask, voxel_size_isotrophic=voxel_size_isotrophic, gpu=gpu_index) 

   elif tissue=="Air, Dry (near sea level)":
      density_vol =  mask *  get_rho_str(ct_volume_in_hu, cparam("ATT_AIR"))
      return density_vol, kvp_array, final_spectrum, att_tissue,  get_projections_new(density_vol, voxel_size_isotrophic=voxel_size_isotrophic, gpu=gpu_index), get_projections_new(mask, voxel_size_isotrophic=voxel_size_isotrophic, gpu=gpu_index) 


def save_hdf5_projections(filepath, radiographs, lungthicknesses): 
   """
      Function to save a compressed hdf5 file

      Args:
         filepath: the desired location
         volume: a 3d numpy volume.

   """
   assert("projections" in filepath) # just be ensure we do not overwrite something important
   with h5py.File(filepath, "w") as h5f:
      for i in range(0, len(radiographs)):
         dset = h5f.create_dataset("radiograph_"+str(i), data=radiographs[i].astype(np.float32), compression="gzip")
         dset2 = h5f.create_dataset("lung_"+str(i), data=lungthicknesses[i].astype(np.float32), compression="gzip")


def get_projections_polychromatic(ctscan, voxel_size_isotrophic, kvp=80, verbose=False,gpu_index=1):
    if verbose:
        print("Get Bone Projections")
    
    density_vol, kvp_array, final_spectrum, mass_att_bone,  rhomap_bone, distance_map = get_rho_map(kvp, "Bone, Cortical (ICRP)",  ctscan, ctscan>cparam("HU_BONE_START"),  voxel_size_isotrophic=voxel_size_isotrophic, gpu_index=gpu_index)

    if verbose:
        print("Get Soft Tissue Projections")
        
    density_vol2, kvp_array, final_spectrum, mass_att_tissue,  rhomap_tissue, distance_map2 = get_rho_map(kvp, "Tissue, Soft (ICRP)",  ctscan, (ctscan<cparam("HU_SOTTISSUE_RANGE")[1]) & (ctscan>cparam("HU_SOTTISSUE_RANGE")[0]),  voxel_size_isotrophic=voxel_size_isotrophic, gpu_index=gpu_index)
  
    if verbose:
        print("Get Adipose Tissue Projections")
    
    density_vol3, kvp_array, final_spectrum, mass_att_adipose,  rhomap_adipose, distance_map3 = get_rho_map(kvp, "Adipose Tissue (ICRP)",  ctscan, (ctscan<cparam("HU_ADIPOSE_TISSUE_RANGE")[1]) & (ctscan>cparam("HU_ADIPOSE_TISSUE_RANGE")[0]),  voxel_size_isotrophic=voxel_size_isotrophic, gpu_index=gpu_index)
    if verbose:
        print("Get Air ")
       
    if verbose:
        print("Done with Projections")
    
    voxel_size_along_x_axis = voxel_size_isotrophic # use isotropic voxels
    voxel_size_along_x_axis_in_cm2 = voxel_size_isotrophic/10 # as densities are in g/cm3, we need to use cm instead of mm

    result = np.zeros_like(rhomap_bone)
    flatfield_exp = 0
    for i in range(0,len(final_spectrum)): # iterate over keV values of spectrum
        exponent =  np.exp(-mass_att_bone[i] * rhomap_bone * voxel_size_along_x_axis_in_cm2  - mass_att_tissue[i] * rhomap_tissue * voxel_size_along_x_axis_in_cm2  - mass_att_adipose[i] * rhomap_adipose * voxel_size_along_x_axis_in_cm2)  
        result+= kvp_array[i] * final_spectrum[i] * exponent
        flatfield_exp += kvp_array[i] * final_spectrum[i] * 1 # exp(0) = 1

    return -np.log(result/flatfield_exp)
