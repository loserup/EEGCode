# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:12:01 2018

Reference: https://github.com/pbashivan/EEGLearn

@author: Long
"""

import scipy
import math as m
import numpy as np
import matplotlib.pyplot as plt

### 第一步：将脑电极3D坐标转换成2D的
locs_3d = scipy.io.loadmat('locs.mat')['locs'] # 在matlab中该变量名为locs 
locs_2d = []

# 将脑电极3d坐标转换为2d的功能函数
def azim_proj(pos):
    """azim_proj : Computes the Azimuthal Equidistant Projection of input
    point in 3D Cartesian Coordinates. Imagine a plane being placed against
    (tangent to) a globe. If a light source inside the globe projects the
    graticule onto the plane the result would be a planar, or azimuthal, map
    projection.

    Parameters:
    -----------
    - pos: position in 3D Cartesian coordinates

    Returns:
    --------
    - projected coordinates using Azimuthal Equidistant Projection
    """

    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])

    return pol2cart(az, m.pi / 2 - elev)

def cart2sph(x, y, z):
    """cart2sph: transform Cartesian coordinates to spherical

    Parameters:
    -----------
    - x: X coordinate
    - y: Y coordinate
    - z: Z coordinate

    Returns:
    - radius, elevation, azimuth
    """

    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)  # r
    elev = m.atan2(z, m.sqrt(x2_y2))  # Elevation
    az = m.atan2(y, x)  # Azimuth

    return r, elev, az

def pol2cart(theta, rho):
    """pol2cart: transform polar coordinates to Cartesian

    Parameters:
    -----------
    - theta: angle value
    - rho: radius value

    Returns:
    --------
    - X, Y
    """

    return rho * m.cos(theta), rho * m.sin(theta)

rotation = np.array([[0,1,0],[-1,0,0],[0,0,1]]) # 逆时针旋转90°矩阵
# 因为原数据x轴正方向指向颈背，y轴正方向指向右耳，需逆时针旋转90°
# 变成x轴正方向指向右耳，y轴正方向指向鼻子
locs_3d = np.dot(locs_3d,rotation) 
# 将脑电极3d坐标转换为2d
for e in locs_3d:
    locs_2d.append(azim_proj(e))
    
#scipy.io.savemat('test.mat', {'test' : locs_2d})
#x,y = [],[]
#for i in range(len(locs_2d)):
#    x.append(locs_2d[i][0])
#    y.append(locs_2d[i][1])
#plt.scatter(x,y)
    
### 以上是第一步：将脑电极3D坐标转换成2D的

### 第二步：将EEG特征矩阵和脑电极2D坐标做成多通道图像
from scipy.interpolate import griddata   
from sklearn.preprocessing import scale 

def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1,
               n_components=2, edgeless=False):
    """gen_images : Generates EEG images given electrode locations in 2D space
    and multiple feature values for each electrode

    Parameters:
    -----------
    - locs: 一个尺寸为[n_electrodes, 2]的数组，包含每个电极的X,Y坐标
    - features: Feature matrix as [n_samples, n_features]
        features are as columns.
        Features corresponding to each frequency band are concatenated.
        (alpha1, alpha2, ..., beta1, beta2,...)
    - n_gridpoints: Number of pixels in the output images
    - normalize: Flag for whether to normalize each band over all samples
    - augment: Flag for generating augmented images
    - pca: Flag for PCA based data augmentation
    - std_mult: Multiplier for std of added noise
    - n_components: Number of components in PCA to retain for augmentation
    - edgeless: If True generates edgeless images by adding artificial channels
        at four corners of the image with value = 0 (default=False).

    Return:
    -------
    - Tensor of size [samples, colors, W, H] containing generated images.
    """

    feat_array_temp = []
    nElectrodes = locs.shape[0]  # Number of electrodes

    # Test whether the feature vector length is divisible by
    # number of electrodes
    assert features.shape[1] % nElectrodes == 0 # 输入特征维数必须能被电极数整除

    n_colors = features.shape[1] / nElectrodes # 颜色数即频带数，EEGLearn有3种频带
    
    # 每个频带的特征分别做为列表feat_array_temp的一个元素
    for c in range(int(n_colors)):
        feat_array_temp.append(
            features[:, c * nElectrodes: nElectrodes * (c + 1)])

    """
    # augment
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(
                    feat_array_temp[c], std_mult,
                    pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(
                    feat_array_temp[c], std_mult,
                    pca=False, n_components=n_components)
    """
    
    nSamples = features.shape[0] # trail数

    # Interpolate the values
    # grid_x在维面上的取值从min(locs[:, 0])到max(locs[:, 0])，共n_gridpoints个点
    grid_x, grid_y = np.mgrid[
        min(locs[:, 0]):max(locs[:, 0]):n_gridpoints * 1j,
        min(locs[:, 1]):max(locs[:, 1]):n_gridpoints * 1j
    ]

    temp_interp = []
    # EEGLearn得到的temp_interp的shape是(3,2670,32,32),即每个频带里每次trial都对应有一个32x32的image
    for c in range(int(n_colors)):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))

    """
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(
            locs,
            np.array(
                [[min_x, min_y], [min_x, max_y],
                 [max_x, min_y], [max_x, max_y]]), axis=0)

        for c in range(n_colors):
            feat_array_temp[c] = np.append(
                feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    """
    
    # Interpolating
    for i in range(nSamples):
        for c in range(int(n_colors)):
            temp_interp[c][i, :, :] = griddata(
                locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                method='cubic', fill_value=np.nan)

        print('Interpolating {0}/{1}\r'.format(i + 1, nSamples), end='\r')

    # Normalizing
    for c in range(int(n_colors)):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])

    # swap axes to have [samples, colors, W, H]
    return np.swapaxes(np.asarray(temp_interp), 0, 1)

