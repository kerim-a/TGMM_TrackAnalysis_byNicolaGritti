import os
import glob
import numpy as np
from skimage.io import imread
import optireg

master_folder = os.path.join('Y:','/Kerim_Anlas/gastruloid_imaging/tracking_preprocess/preprocess')
folders = [
    os.path.join(master_folder,i) for i in [
        # '2021_PD_72h\\20210325_pos4_72h_reg_g',
        # '2021_PD_72h\\20210429_pos1_72h_reg_g',
        # '2021_PD_72h\\20210429_pos3_72h_reg_g',
        # '2021_PD_72h\\20210429_pos4_72h_top_reg_g',
        # '2021_PD_96h\\20210326_pos3_reg_g',
        # '2021_PD_96h\\20210326_pos4_reg_g',
        # '2021_PD_96h\\20210330_pos1_reg_g',
        # '2021_PD_96h\\20210330_pos2_reg_g',
        # '2021_PD_96h\\20210330_pos3_reg_g',
        # '20211103_ESLmix_48h_good\\Pos2_good_reg_g',
        # '20211110_ESLmix_48h_firstTex_probnochi\\Pos1_good_reg_g',
        # '20211110_ESLmix_48h_firstTex_probnochi\\Pos2_good_reg_g',
        # '20211110_ESLmix_48h_firstTex_probnochi\\Pos3_best_reg_g',
        # '20211117_ESLmix_72h_good\\Pos1_reg_g',
        # '20211117_ESLmix_72h_good\\Pos2_reg_g',
        # '20211117_ESLmix_72h_good\\Pos3_reg_g',
        # '20211117_ESLmix_72h_good\\Pos4_reg_g',
        # '20211118_ESLmix_96h_good\\Pos1_goodfewcells_reg_g', ### too big for gpu
        # '20211118_ESLmix_96h_good\\Pos2_decentfewcells', # not reg
        # '20211118_ESLmix_96h_good\\Pos3_good_reg_g', ### too big for gpu
        # '20211118_ESLmix_96h_good\\Pos4_best_reg_g',
        # '20211125_ESLmix_72h_good\\Pos1_goodbutdense_reg_g',
        # '20211125_ESLmix_72h_good\\Pos4_okbutdense_reg',
        # '20211126_ESLmix_96h_good\\Pos3_good_butonlytpos_reg_g',
        # '20220204_a2imix_96h_LC\\Pos1_reg_g',
        # '20220204_a2imix_96h_LC\\Pos2_reg_g',
        # '20220204_a2imix_96h_LC\\Pos3_reg_g',
        # '20220204_a2imix_96h_LC\\Pos4_reg_g',
        # '20220211_a2imix_96h_good\\Pos1',
        # '20220211_a2imix_96h_good\\Pos2',
        # '20220211_a2imix_96h_good\\Pos3',
        # '20220211_a2imix_96h_good\\Pos4',
        # '20210503-04_PD_IWP3\\72h\\Position_3_Settings_1_ok_d150',
        # '20210503-04_PD_IWP3\\72h\\Position_4_Settings_1_ok_d150',
        # '20210503-04_PD_IWP3\\96h',
        # '20211118_ESLmix_96h_good\\Pos3_good_reg_g',
        #  '20211118_ESLmix_96h_good\\Pos1_goodfewcells_reg_g',
        #  '20210503-04_PD_IWP3\\72h\\test',
           '2021_PD_72h\\20210325_pos3_72h_polvisible',
           '2021_PD_72h\\20210325_pos2_72h',
           '20211103_ESLmix_48h_good\\Pos1_dirtbutgood',
           '20211111_ESLmix_72h_polvisible\\pos1',
           '20211125_ESLmix_72h_good\\Pos2_goodsomewhatpol',
           '2021_PD_96h\\20210430_pos4_early96h',
        ]
]

available_channels = ['mKO2', 'GFP'] # available channels (folders)
pixel_size = (2,0.3467,0.3467) # anistotropic pixel size (ordered: ZYX)
reference_channel = 'mKO2' # channel used to find cells
cell_diameter_XY = 12 # in pixels,previously 10
reverse_order = True # process from last timepoint to first (i.e. register all images relative to last frame)
threshold = 200 # maxima with values lower than threshold will not be considered as cells
gpu_down = 1. # scale image down to use less GPU memory and improve speed
with_cle = True # whether to use pyclesperanto to transform the image

for folder in folders:
    print('*********************')
    print(folder)

    flist = glob.glob(os.path.join(folder, reference_channel, '*.tif'))
    flist.sort()
    # flist = flist[:10]
    print('Found %d images.'%len(flist))

    scale = pixel_size[0]/pixel_size[1]
    print('Pixel size in Z : %.5f um'%pixel_size[0])
    print('Pixel size in XY: %.5f um'%pixel_size[1])
    print('Anisotropy: %.5f'%scale)

    # compute anchor points or load file if it already exists
    df = optireg.find_anchor_points(folder, flist, fileName = 'anchor_points.csv', 
                            scale = scale, gpu_down = gpu_down, 
                            cell_diameter_XY = cell_diameter_XY,
                            reverse_order = reverse_order,
                            threshold = threshold)

    tps = list(set(df.tpreg))
    tps.sort()
    df_list = [df[df.tpreg==tp] for tp in tps]

    Ts = optireg.find_transformations(df_list, folder, maxiter=100, tol=1e-5)

    df_list = optireg.perform_points_registration(df_list, Ts)

    # from importlib import reload

    # reload(optireg)
    optireg.perform_images_registration(folder, 
                                        available_channels, 
                                        reverse_order, 
                                        scale,
                                        Ts,
                                        with_cle = with_cle,
                                        registered_folder_id = '_reg1')

