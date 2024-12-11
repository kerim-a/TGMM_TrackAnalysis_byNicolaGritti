import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import (
    asin, pi, atan2, cos, sin
)


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    # if np.linalg.det(R) < 0:
    #    Vt[m-1,:] *= -1
    #    R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(source, destination, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert source.shape == destination.shape

    # get number of dimensions
    m = source.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,source.shape[0]))
    dst = np.ones((m+1,destination.shape[0]))
    src[:m,:] = np.copy(source.T)
    dst[:m,:] = np.copy(destination.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
    prev_error = np.mean(distances)
    
    error_history = [prev_error]

    for i in range(max_iterations):
        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        
        error_history.append(mean_error)
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(source, src[:m,:].T)

    return T, distances, i, error_history


# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:12:48 2019

@author: Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript: Functions for transforming point clouds or images using
           homogeneous transformation matrices such as those returned
           by ICP (github.com/ClayFlannigan/icp/blob/master/icp.py).
            
@disclaimer: Licensed under MIT, copyright 2019 Jonas Hartmann
             Code provided "as is", without warranty of any kind.
"""

import numpy as np
import scipy.ndimage as ndi

def cloud_transform(C, TM):
    """Transform an m-dimensional point cloud `C` by the  
    homogeneous transformation matrix `TM`.
    
    C  : numpy array of shape (n, m), where 
           n is the number of points
           m is the number of dimensions
    TM : numpy array of shape ((m+1), (m+1))
    
    -> CT : transformed cloud, 
            numpy array of shape (n, m)
    """

    CP = np.ones((C.shape[1]+1, C.shape[0]))
    CP[:C.shape[1], :] = np.copy(C.T)
    CP = np.dot(TM, CP)
    CT = CP[:C.shape[1], :].T

    return CT

def image_transform(im, TM, origin=None, output_shape=None, order=1):
    """Transform an m-dimensional image `im` by the  
    homogeneous transformation matrix `TM`.
    
    im : m-dimensional numpy image array
    TM : numpy array of shape ((m+1), (m+1))
    
    origin : numpy array of shape (m,), default None
             coordinates (in image space) of the origin 
             of the coordinate system within which TM 
             is defined. Use None if the TM is defined 
             in the same coordinate system as the image.
    output_shape : tuple of ints, default None
             Shape of the output image. If None, it has
             the same shape as the input.
    
    -> imT : transformed image, 
             m-dimensional numpy image array
    """

    # Get the linear transformation matrix
    A_L = TM[:-1,:-1]
    c = 0.0
    
    # Get offset to handle coordinate system translation
    if origin is not None:
        x_0 = origin
        c_L = TM[:-1,-1]
        c   = - np.dot(A_L, x_0) + x_0 + c_L        

    # Transform the image
    imT = ndi.affine_transform(im, A_L, offset=c, 
                               output_shape=output_shape,
                               order=order)
    
    return imT

"""
from:
https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix#:~:text=If%20R%20is%20the%20(3x3,sum%20of%20the%20diagonal%20elements).

Illustration of the rotation matrix / sometimes called 'orientation' matrix
R = [ 
       R11 , R12 , R13, 
       R21 , R22 , R23,
       R31 , R32 , R33  
    ]

REMARKS: 
1. this implementation is meant to make the mathematics easy to be deciphered
from the script, not so much on 'optimized' code. 
You can then optimize it to your own style. 

2. I have utilized naval rigid body terminology here whereby; 
2.1 roll -> rotation about x-axis 
2.2 pitch -> rotation about the y-axis 
2.3 yaw -> rotation about the z-axis (this is pointing 'upwards') 
"""

def decompose_rot(T):
    a = T[:-1,:-1]
    if a[2,0] != 1 and a[2,0] != -1: 
         pitch_1 = -1*asin(a[2,0])
         pitch_2 = pi - pitch_1 
         roll_1 = atan2( a[2,1] / cos(pitch_1) , a[2,2] /cos(pitch_1) ) 
         roll_2 = atan2( a[2,1] / cos(pitch_2) , a[2,2] /cos(pitch_2) ) 
         yaw_1 = atan2( a[1,0] / cos(pitch_1) , a[0,0] / cos(pitch_1) )
         yaw_2 = atan2( a[1,0] / cos(pitch_2) , a[0,0] / cos(pitch_2) ) 

         # IMPORTANT NOTE here, there is more than one solution but we choose the first for this case for simplicity !
         # You can insert your own domain logic here on how to handle both solutions appropriately (see the reference publication link for more info). 
         pitch = pitch_1 
         roll = roll_1
         yaw = yaw_1 
    else: 
         yaw = 0 # anything (we default this to zero)
         if a[2,0] == -1: 
            pitch = pi/2 
            roll = yaw + atan2(a[0,1],a[0,2]) 
         else: 
            pitch = -pi/2 
            roll = -1*yaw + atan2(-1*a[0,1],-1*a[0,2]) 

    # convert from radians to degrees
#     roll = roll*180/pi 
#     pitch = pitch*180/pi
#     yaw = yaw*180/pi 
    yaw = 0.
    pitch = 0.

    rxyz_deg = [roll , pitch , yaw]

    yawMatrix = np.matrix([
    [cos(yaw), -sin(yaw), 0],
    [sin(yaw), cos(yaw), 0],
    [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
    [cos(pitch), 0, sin(pitch)],
    [0, 1, 0],
    [-sin(pitch), 0, cos(pitch)]
    ])

    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, cos(roll), -sin(roll)],
    [0, sin(roll), cos(roll)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix
    
    T[:-1,:-1] = R

    # remove z drift
    T[0,3] = 0.

    return np.array(T)

############################################################

# from skimage.io import imread, imsave
# import numpy as np
# import pandas as pd

# # a = imread('ch2_ang000.tif')
# # b = imread('ch2_ang180.tif')

# p = pd.read_csv('landmarks.csv')

# print(p)

# a = p.iloc[::2]
# a = np.array(a[['X','Y','Slice']])
# b = p.iloc[1::2]
# b = np.array(b[['X','Y','Slice']])

# print(a)
# print(b)
# # b = 

# print('computing icp')
# T, d, i = icp(a,b)
# print(T)

# print('reading images')
# a = imread('ch2_ang000.tif')
# b = imread('ch2_ang180.tif')

# print('tranforming image b')
# b = image_transform(b,T)
# imsave('ch2_ang180_2.tif', b.astype(np.uint16))

# print('tranforming image a')
# a = image_transform(a,T)
# imsave('ch2_ang180_2.tif', a.astype(np.uint16))

