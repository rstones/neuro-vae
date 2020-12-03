#!/usr/bin/env python
import numpy as np

def Calc_Dims_3d(shape, padding, kernel, stride, pool_kernel, pool_stride):
    convshape = np.ceil((shape + 2*padding - kernel) / stride)
    poolshape = np.ceil((convshape - pool_kernel) / pool_stride)
    
    return poolshape

