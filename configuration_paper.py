#@author M. Schultheiss, T. Dorosti
CONFIGURATIONS = {
        "256_DAX" : { 
    
        # Target X-ray spectrum
        "KVP": 70,
        "IMAGEWIDTH": 256,
        "HU_BONE_START": 240,
        "HU_SOTTISSUE_RANGE": [0,240],
        "HU_ADIPOSE_TISSUE_RANGE": [-200, 0],

        # Approximate attenuation of tissues of CT scanner spectrum
        "ATT_WATER": 0.19285246,
        "ATT_BONE":0.25487029,
        "ATT_SOFTTISSUE":0.19059642,
        "ATT_ADIPOSE": 0.18795962,
        
        # X-Ray properties
        "SOURCE_TO_DETECTOR": 2435, # in mm 
        "PATIENT_TO_DETECTOR":  375, # in mm 
        "DETECTOR_AL": 2.8, # in mm
        "DETECTOR_CSI_TICKNESS": 0.6, # in mm 
        "DETECTOR_CSI_DENSITY": 3.383,  # in g/cm3
        "DETECTOR_HEIGHT": 400, # in mm
        "DETECTOR_WIDTH": 400,  # in mm
        }
}

CURRENT_CONFIG = "256_DAX"
POSITION = 'pa' # 'pa' for frontal radiographs, 'lat' for lateral radiographs
PATH_SERVER_CACHE = '???' # general path to save simulated radiographs
CURRENT_GPU  = '???' 

def cparam(param):
    return CONFIGURATIONS[CURRENT_CONFIG][param]
