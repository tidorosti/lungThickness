"""
    Attenuation functions partly taken from the chairs pyCT library
"""

import re
from massdata.molarmasses import molar_masses
import xraylib as xr
import numpy as np


def nist_string_to_dict(s, do_norm=True):
    """\
    Segment material string `s` into elements with respective ratios 

    Parameters:
    -----------
    s : string
        Definition of material. 
        Can be compound ('H2O') or mixture ('C0.5H0.3N0.2').
    do_norm : bool, optional
        Flag to perform normalization to sum(values) = 1
        Default is True

    """

    # list of all components in `s` (split at capital letters)
    cmp_lst = re.findall('[A-Z][^A-Z]*', s)

    # dict to be returned
    d = {}

    # `s` is mixture
    if '.' in s:
        for c in cmp_lst:
            zero_index = c.find('.') - 1
            mat = c[:zero_index]
            ratio = float(c[zero_index:])
            d[mat] = ratio

    # `s` is compound
    else:
        # loop over all elements in cmp_lst
        for elem in cmp_lst:
            el_str = ""
            num_str = ""
            # split `elem` into element name and number
            for c in elem:
                if c.isalpha():
                    el_str += c
                else:
                    num_str += c

            # set `num_str` to 1 if no number in `elem`
            if num_str == "": num_str = 1

            # add element with number to return dict
            d[el_str] = int(num_str)

    if do_norm:
        ratio_sum = float( sum(d.values()) )
        d = {mat: ratio / ratio_sum for mat, ratio in d.items()}

    return d

def get_attenuation(materials, energy):

    if not hasattr(materials, '__iter__'):
        materials = [materials]

    f_atten = [np.vectorize(xr.CS_Total_CP) if isinstance(mat, str)
               else np.vectorize(xr.CS_Total)
               for mat in materials]

    return np.array([f(materials[i], energy) for i, f in enumerate(f_atten)])