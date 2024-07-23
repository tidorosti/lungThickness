#@author M. Schultheiss

import numpy as np
from os.path import isfile, join
from os import listdir
import os
import csv

def _load_data_csv_labels(csv_path: str) -> None:
    """
        Load CSV file to list
    """
    with open(csv_path, 'r') as csvfile:
        data_csv = csv.reader(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        return list(data_csv)


class Slice2D(np.ndarray):

    def __new__(cls, input_array, class_label=None, scale="HU", filename=None, voxel_size=[1,1]):

        if input_array.ndim != 2 and input_array.ndim != 3: # allow [x,y, 1] here...
            raise ValueError("Array must have 2 dimensions (x, y).")

        obj = np.asarray(input_array).view(cls)
        obj.scale = scale
        obj.voxel_size = voxel_size
        obj.filename = filename
        obj.class_label = class_label
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.scale = getattr(obj, 'scale', None)
        self.class_label = getattr(obj, 'class_label', None)
        self.filename = getattr(obj, 'filename', None)
        self.voxel_size = getattr(obj, 'voxel_size', None)

class Slice2DSet(list):

    def __init__(self, *args):
        self.log = []
        list.__init__(self, *args)

    def get_different_filenames(self):
        filenames = []
        for i in range(0, len(self)):
            if self[i].filename not in filenames:
                filenames.append(self[i].filename)

    def split_data_by_csv(self, prefix: str):
        """
            Split slice2dset given a path to a folder with train, test and validation csv list.

            Args:
                slice2dset: A Slice 2D Set
                prefix: Path to the folder with the CSV lists. CSV list must 
                contain a filename in each row. 
            Returns:
                trainX, trainY, testX, testY, valX, valY
        """
        trainX, trainY = [], []
        testX, testY = [], []
        valX, valY = [], []

        fntrain = [fn[0] for fn in _load_data_csv_labels(prefix+"_train.csv")]
        fntest = [fn[0] for fn in _load_data_csv_labels(prefix+"_test.csv")]
        fnval = [fn[0] for fn in _load_data_csv_labels(prefix+"_val.csv")]

        for slice2d in self:
            if slice2d.filename in fntrain:
                trainX.append(slice2d[:])
                trainY.append(slice2d.class_label[:])
            elif slice2d.filename in fntest:
                testX.append(slice2d[:])
                testY.append(slice2d.class_label[:])
            elif slice2d.filename in fnval:
                valX.append(slice2d[:])
                valY.append(slice2d.class_label[:])
        return trainX, trainY, testX, testY, valX, valY

