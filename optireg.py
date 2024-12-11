
import os, re, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imread, imsave
import pyclesperanto_prototype as cle
from scipy.spatial.transform import Rotation
from scipy import optimize
from register_icp import cloud_transform, image_transform
from sklearn.neighbors import NearestNeighbors
import subprocess as sp

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

'''
Process images to find most of the cells
'''
def find_anchor_points(folder, flist, fileName = 'anchor_points.csv', 
                        scale = 1, gpu_down = 1., cell_diameter_XY = 10, threshold = 200,
                        reverse_order = True):

    anchor_file = os.path.join(folder, fileName)

    if not os.path.exists(anchor_file):
        df = pd.DataFrame({})

        for i in tqdm(range(len(flist)), total=len(flist)):
            fname = flist[i]

            img_input = imread(fname)
            
            img = cle.create([int((img_input.shape[0] * scale)/gpu_down),
                            int(img_input.shape[1]/gpu_down),
                            int(img_input.shape[2]/gpu_down)])
            cle.scale(img_input, img,
                    factor_x = 1./gpu_down,
                    factor_y = 1./gpu_down,
                    factor_z = scale/gpu_down,
                    centered = False,
                    linear_interpolation = True)
            del img_input
            
    #         print('Gaussian')
            sigmaxy = cell_diameter_XY/gpu_down
            sigma = [sigmaxy,sigmaxy,sigmaxy]
            
            img_gauss = cle.gaussian_blur(img, sigma_x=sigma[2], sigma_y=sigma[1], sigma_z=sigma[0])
            del img

    #         print('Find maxima')
            detected_spots = cle.detect_maxima_box(img_gauss, 
                                                radius_x=0,#sigma[2]/2, 
                                                radius_y=0,#sigma[1]/2, 
                                                radius_z=0,)#)sigma[0]/2)
            selected_spots = cle.binary_and(img_gauss>threshold, detected_spots)
            del img_gauss
            p = np.where(selected_spots)
    #         p = peak_local_max(img_gauss, threshold_abs=200, min_distance=5)

            df1 = pd.DataFrame({
                                'tp': int(re.findall(r"t(\d{4})_", " "+fname+ " ")[0])-1,
                                'z': p[0].astype(float)*gpu_down,
                                'y': p[1].astype(float)*gpu_down,
                                'x': p[2].astype(float)*gpu_down,
                                })
            df = pd.concat([df,df1], ignore_index=True)
        
        if reverse_order:
            df.loc[:,'tpreg'] = np.max(df.tp)-df.tp
        else:
            df.loc[:,'tpreg'] = df.tp

        df.to_csv(anchor_file, index=False)

    else:
        df = pd.read_csv(anchor_file)

    return df

########################
'''
Find registration via mean distance minimization
'''

def transformation(params):
    angles = params[:3]
    translation = params[3:]
    rot = Rotation.from_rotvec(angles).as_matrix()
    
    # special reflection case
    if np.linalg.det(rot) < 0:
        [U,S,V] = np.linalg.svd(rot)
        # multiply 3rd column of V by -1
        rot = V * U.T
    
    mat = np.eye(4)
    mat[:3,:3] = rot
    mat[:3,3] = translation
    
    return mat

def loss(params, points_ant, points_now):

    points_now_new = cloud_transform(points_now,transformation(params))
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(points_now_new)
    distances, indices = nbrs.kneighbors(points_ant)

    return np.mean(distances)

def find_transformations(df_list, folder, fileName = 'transformations.npz', maxiter=100, tol=1e-5):

    transformation_name = os.path.join(folder, fileName)

    if not os.path.exists(transformation_name):

        Ts = [np.eye(4) for i in range(len(df_list))]

        for i in tqdm(range(len(df_list)-1)):

            pos_ant = df_list[i][['z','y','x']].to_numpy()
            pos_now = df_list[i+1][['z','y','x']].to_numpy()

            initial_guess = [0,0,0,0,0,0]

            result = optimize.minimize(loss, initial_guess, 
                                    args = (pos_ant, pos_now), 
                                    options={'maxiter': maxiter}, tol=tol)

            Ts[i+1] = transformation(result.x)

        np.savez(transformation_name, Ts=Ts)

    else:

        Ts = np.load(transformation_name)['Ts']

    return Ts

'''
Apply transformations
'''

def perform_points_registration(df_list, Ts):

    for i in range(len(df_list)):
        df_list[i].loc[:,'xreg'] = np.array(df_list[i]['x'])
        df_list[i].loc[:,'yreg'] = np.array(df_list[i]['y'])
        df_list[i].loc[:,'zreg'] = np.array(df_list[i]['z'])
    
    for i in range(len(df_list)):

        A = df_list[i][['z','y','x']].to_numpy()

        T = np.eye(4)
        for j in range(i,0,-1):
            T = np.dot(Ts[j],T)
        A = cloud_transform(A, T)

        df_list[i].loc[:,'zreg'] = A[:,0]
        df_list[i].loc[:,'yreg'] = A[:,1]
        df_list[i].loc[:,'xreg'] = A[:,2]

    return df_list

def perform_images_registration(folder, available_channels, reverse_order, scale, Ts, with_cle=False, registered_folder_id='_reg'):

    for channel in available_channels:
        print(channel)

        infolder = os.path.join(folder, channel)
        flist = glob.glob(os.path.join(infolder, '*.tif'))
        flist.sort()
        if reverse_order:
            flist = flist[::-1]

        outfolder = os.path.join(folder,channel+registered_folder_id)

        img = imread(flist[0])
        shape = np.array([img.shape[0], img.shape[1], img.shape[2]])
        pad = (shape*0.0).astype(int)

        if not os.path.exists(os.path.join(outfolder)):
                os.mkdir(os.path.join(outfolder))
        for mip_folder in ['MIP_XY','MIP_YZ','MIP_XZ']:
            if not os.path.exists(os.path.join(outfolder,mip_folder)):
                os.mkdir(os.path.join(outfolder,mip_folder))
        for mip_folder in ['MIP_XY','MIP_YZ','MIP_XZ']:
            if not os.path.exists(os.path.join(infolder,mip_folder)):
                os.mkdir(os.path.join(infolder,mip_folder))

        i=0
        for f in tqdm(flist, total=len(flist)):
                
            fname = os.path.split(f)[-1]
            if reverse_order:
                fname_reg = 'tp%04d_'%(len(flist)-i-1)+channel+'.tif'
            else:
                fname_reg = 'tp%04d_'%(i)+channel+'.tif'

    #         if not os.path.exists(os.path.join(outfolder,'t%04d.tif'%(i+1))):

            img_input = imread(f)
        
            # save max projection of original data
            if not os.path.exists(os.path.join(infolder,'MIP_XY',fname)):
                imsave(os.path.join(infolder,'MIP_XY',fname),
                    np.max(img_input.astype(np.uint16),0),
                    check_contrast=False)
            if not os.path.exists(os.path.join(infolder,'MIP_XZ',fname)):
                imsave(os.path.join(infolder,'MIP_XZ',fname),
                    np.max(img_input.astype(np.uint16),1),
                    check_contrast=False)
            if not os.path.exists(os.path.join(infolder,'MIP_YZ',fname)):
                imsave(os.path.join(infolder,'MIP_YZ',fname),
                    np.max(img_input.astype(np.uint16),2),
                    check_contrast=False)
            
            #make isotropic
            # print('1',get_gpu_memory()[0])

            img_iso = cle.create([int(img_input.shape[0] * scale), 
                                    int(img_input.shape[1]), 
                                    int(img_input.shape[2])])
            # print('2',get_gpu_memory()[0])
            cle.scale(img_input, img_iso, 
                        factor_x=1, factor_y=1, factor_z=scale, centered=False,
                        linear_interpolation=True)
            del img_input
            # print('3',get_gpu_memory()[0])

            img_iso = np.pad(img_iso, [[p,p] for p in pad])

            # find out transformation
            T = np.eye(4)
            for j in range(i,0,-1):
                T = np.dot(Ts[j],T)

            # compute transformation
            if with_cle:
                # print('4',get_gpu_memory()[0])
                img_iso = cle.transpose_xz(img_iso)
                # print('5',get_gpu_memory()[0])
                img_reg_iso = cle.affine_transform(img_iso, transform=T, 
                                            linear_interpolation=True)
                # print('6',get_gpu_memory()[0])
                del img_iso
                # print('7',get_gpu_memory()[0])
                img_reg_iso = cle.transpose_xz(img_reg_iso)
                # print('8',get_gpu_memory()[0])

                ### this is very slow
                # img = np.swapaxes(img, 0, 2)
                # img_reg_iso = cle.affine_transform(img, transform=T)
                # img_reg_iso = np.swapaxes(img_reg_iso, 0, 2)
            else:
                img_reg_iso = image_transform(img_iso,np.linalg.inv(T),order=1,origin=pad)
                del img_iso
            # print('9',get_gpu_memory()[0])
            img_reg = cle.create([int(img_reg_iso.shape[0] / scale), 
                                    int(img_reg_iso.shape[1]), 
                                    int(img_reg_iso.shape[2])])
            # print(get_gpu_memory()[0])
            cle.scale(img_reg_iso, img_reg, 
                        factor_x=1, factor_y=1, factor_z=1/scale, centered=False,
                        linear_interpolation=True)
            # print(get_gpu_memory()[0])
            
            imsave(os.path.join(outfolder,fname_reg),
                img_reg.astype(np.uint16),
                check_contrast=False)

            imsave(os.path.join(outfolder,'MIP_XY',fname_reg),
                np.max(img_reg,0).astype(np.uint16),
                check_contrast=False)
            imsave(os.path.join(outfolder,'MIP_XZ',fname_reg),
                np.max(img_reg,1).astype(np.uint16),
                check_contrast=False)
            imsave(os.path.join(outfolder,'MIP_YZ',fname_reg),
                np.max(img_reg,2).astype(np.uint16),
                check_contrast=False)

            del img_reg_iso, img_reg

            i += 1