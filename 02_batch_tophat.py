import os
import glob
import numpy as np
from skimage.io import imread, imsave
import time
from tqdm import tqdm
import pyclesperanto_prototype as cle

master_folder = os.path.join('Y:','/Kerim_Anlas/gastruloid_imaging/tracking_preprocess/preprocess')
folders = [
    os.path.join(master_folder,i) for i in [
        #'2021_PD_72h\\20210325_pos4_72h_reg_g',
        #'2021_PD_72h\\20210429_pos1_72h_reg_g',
        #'2021_PD_72h\\20210429_pos3_72h_reg_g',
        #'2021_PD_72h\\20210429_pos4_72h_top_reg_g',
        #'2021_PD_96h\\20210326_pos3_reg_g',
        #'2021_PD_96h\\20210326_pos4_reg_g',
        #'2021_PD_96h\\20210330_pos1_reg_g',
        #'2021_PD_96h\\20210330_pos2_reg_g',
        #'2021_PD_96h\\20210330_pos3_reg_g',
        #'20211103_ESLmix_48h_good\\Pos2_good_reg_g',
        #'20211110_ESLmix_48h_firstTex_probnochi\\Pos1_good_reg_g',
        #'20211110_ESLmix_48h_firstTex_probnochi\\Pos2_good_reg_g',
        #'20211110_ESLmix_48h_firstTex_probnochi\\Pos3_best_reg_g',
        #'20211117_ESLmix_72h_good\\Pos1_reg_g',
        #'20211117_ESLmix_72h_good\\Pos2_reg_g',
        #'20211117_ESLmix_72h_good\\Pos3_reg_g',
        #'20211117_ESLmix_72h_good\\Pos4_reg_g',
        # '20211118_ESLmix_96h_good\\Pos1_goodfewcells_reg_g', ### too big for gpu
        # '20211118_ESLmix_96h_good\\Pos2_decentfewcells', # not reg
        # '20211118_ESLmix_96h_good\\Pos3_good_reg_g', ### too big for gpu
        #'20211118_ESLmix_96h_good\\Pos4_best_reg_g',
        #'20211125_ESLmix_72h_good\\Pos1_goodbutdense_reg_g',
        #'20211125_ESLmix_72h_good\\Pos4_okbutdense_reg',
        #'20211126_ESLmix_96h_good\\Pos3_good_butonlytpos_reg_g',
        #'20220204_a2imix_96h_LC\\Pos1_reg_g',
        #'20220204_a2imix_96h_LC\\Pos2_reg_g',
        #'20220204_a2imix_96h_LC\\Pos3_reg_g',
        #'20220204_a2imix_96h_LC\\Pos4_reg_g',
        #'20220211_a2imix_96h_good\\Pos1',
        #'20220211_a2imix_96h_good\\Pos2',
        #'20220211_a2imix_96h_good\\Pos3',
        #'20220211_a2imix_96h_good\\Pos4',
        #'20210503-04_PD_IWP3\\72h\\Position_3_Settings_1_ok_d150',
        #'20210503-04_PD_IWP3\\72h\\Position_4_Settings_1_ok_d150',
        #'20210503-04_PD_IWP3\\96h',
        #'20211118_ESLmix_96h_good\\Pos3_good_reg_g',
        #'20211118_ESLmix_96h_good\\Pos1_goodfewcells_reg_g',
         '2021_PD_72h\\20210325_pos3_72h_polvisible',
         '2021_PD_72h\\20210325_pos2_72h',
         '20211103_ESLmix_48h_good\\Pos1_dirtbutgood',
         '20211111_ESLmix_72h_polvisible\\pos1',
         '20211125_ESLmix_72h_good\\Pos2_goodsomewhatpol',
         '2021_PD_96h\\20210430_pos4_early96h',
        ]
]

img_folders = [os.path.join(i, 'mKO2_reg1') for i in folders]
scale = (2,0.3467,0.3467)
cell_diameter_XY = 12 # in pixels
sigma = (0.2,2,2) # sigma of gaussian blur
make_gauss = False

for img_folder in img_folders:
    print('*******************')
    print(img_folder)
    flist = glob.glob(os.path.join(img_folder,'*.tif'))
    flist.sort()
    print('Found %d images.'%len(flist))

    folder_name = os.path.split(img_folder)[-1]
    tophat_folder = os.path.join(os.path.split(img_folder)[0],folder_name+'_tophat')
    gauss_folder = os.path.join(os.path.split(img_folder)[0],folder_name+'_tophat_gauss')

    if not os.path.exists(tophat_folder):
        os.mkdir(tophat_folder)
    if make_gauss:
        if not os.path.exists(gauss_folder):
            os.mkdir(gauss_folder)

    i=0
    for f in tqdm(flist):
        start = time.time()
        
        img0 = imread(f)
        img0[img0<100] = 100
        load_time = time.time()
        
        img1 = cle.top_hat_box(img0, np.zeros(img0.shape), 
                                radius_z=np.round(4*cell_diameter_XY*scale[1]/scale[0]), 
                                radius_x=np.round(2*cell_diameter_XY), 
                                radius_y=np.round(2*cell_diameter_XY))
        tophat_time = time.time()

        if make_gauss:
            img2 = cle.gaussian_blur(img1, 
                                    sigma_z=sigma[0], 
                                    sigma_x=sigma[1], 
                                    sigma_y=sigma[2])
        gauss_time = time.time()
        
        img1 = img1.astype(np.uint16)
        if make_gauss:
            img2 = img2.astype(np.uint16)
        
        imsave(os.path.join(tophat_folder,'t%04d.tif'%i), 
            cle.pull(img1), 
            check_contrast=False)
        save_tophat_time = time.time()

        if make_gauss:
            imsave(os.path.join(gauss_folder,'t%04d.tif'%i), 
                cle.pull(img2), 
                check_contrast=False)
        save_gauss_time = time.time()

        # print('TP %d: %.4f'%(i, load_time-start), 
        #     '%.4f'%(tophat_time-load_time), 
        #     '%.4f'%(gauss_time-tophat_time),
        #     '%.4f'%(save_tophat_time-gauss_time),
        #     '%.4f'%(save_gauss_time-save_tophat_time))
        
        i+=1
        