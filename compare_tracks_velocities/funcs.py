# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:57:54 2022

@author: gritti
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5)
from collections import Counter
import itertools

from matplotlib import rc
rc('font', size=12)
rc('font', family='Arial')
# rc('font', serif='Times')
rc('pdf', fonttype=42)

##################################

def flatten_list(regular_list):
    '''Falten a 2d list:
    https://stackabuse.com/python-how-to-flatten-list-of-lists/
    '''
    flat_list = list(itertools.chain(*regular_list))
    return flat_list

##################################
def load_df_vertices(fname):
    
    print('Loading vertices csv...')
    
    df_orig = pd.read_csv(fname, header=[0,1,2], encoding='iso-8859-1')
    # print(set([i[0] for i in df_orig.keys()]))
    
    return df_orig

def restructure_df_vertices(df_orig):
    
    print('Restructuring cells dataframe...')
    
    df_cells = pd.DataFrame({})
    # df['label'] = df_orig['Label']['Unnamed: 0_level_1']['Unnamed: 0_level_2']
    df_cells['id'] = df_orig['ID']['Unnamed: 1_level_1']['Unnamed: 1_level_2']
    df_cells['n_links'] = df_orig['Spot N links'][' ']['Unnamed: 2_level_2']
    df_cells['mean-ch1'] = df_orig['Spot intensity']['Mean ch1']['(Counts)']
    df_cells['mean-ch2'] = df_orig['Spot intensity']['Mean ch2']['(Counts)']
    df_cells['x'] = df_orig['Spot position']['X']['(µm)']
    df_cells['y'] = df_orig['Spot position']['Y']['(µm)']
    df_cells['z'] = df_orig['Spot position']['Z']['(µm)']
    df_cells['t'] = df_orig['Spot frame'][' ']['Unnamed: 5_level_2']
    df_cells['r'] = df_orig['Spot radius'][' ']['(µm)']
    df_cells['track_id'] = df_orig['Spot track ID'][' ']['Unnamed: 24_level_2']
    df_cells['track_n_spots'] = df_orig['Track N spots']
    
    return df_cells

##################################
def load_df_edges(fname):
    
    print('Loading edges csv...')
    
    df_orig = pd.read_csv(fname, header=[0,1,2], encoding='iso-8859-1')
    # print(set([i[0] for i in df_orig.keys()]))
    
    return df_orig

def restructure_df_edges(df_orig):
    
    print('Restructuring links dataframe...')

    # sometimes sources and targets are swapped. Check the first values and figure out if they were indeed.
    source0 = df_orig['Link target IDs']['Source spot id']['Unnamed: 3_level_2'].values.astype(int)[0]
    target0 = df_orig['Link target IDs']['Target spot id']['Unnamed: 4_level_2'].values.astype(int)[0]
    are_swapped = source0>target0

    df_links = pd.DataFrame({})
    #df_links['link_frame'] = df_orig['TrackMate Link features']['EDGE_TIME']['(frames)']
    if not are_swapped:
        df_links['source_id'] = df_orig['Link target IDs']['Source spot id']['Unnamed: 3_level_2'].values.astype(int)
        df_links['target_id'] = df_orig['Link target IDs']['Target spot id']['Unnamed: 4_level_2'].values.astype(int)
    else:
        df_links['source_id'] = df_orig['Link target IDs']['Target spot id']['Unnamed: 4_level_2'].values.astype(int)
        df_links['target_id'] = df_orig['Link target IDs']['Source spot id']['Unnamed: 3_level_2'].values.astype(int)
    df_links['speed'] = df_orig['Link velocity'][' ']['(µm/frame)']
    #df_links['directional_change_rate'] = df_orig['TrackMate Link features']['DIRECTIONAL_CHANGE_RATE']['Unnamed: 8_level_2']
    
    #df_links = df_links.sort_values(by=['link_frame','source_id'])
    df_links = df_links.reset_index(drop=True)
    
    return df_links

def append_trackid_to_links(df_cells, df_links):
    
    print('Appending track id to links...')
    
    track_ids = df_cells.track_id.values
    cell_ids = list(df_cells.id.values)
    source_ids = df_links.source_id.values
    
    sorter = np.argsort(cell_ids)
    idxs = sorter[np.searchsorted(cell_ids, source_ids, sorter=sorter)]
    df_links['track_id'] = track_ids[idxs]
    
    return df_links

##################################
def process_csv(vertices = 'FeatureAndTagTable-vertices.csv',
                edges = 'FeatureAndTagTable-edges.csv'):
    
    df_orig = load_df_vertices(vertices)
    df_cells = restructure_df_vertices(df_orig)
    
    df_orig = load_df_edges(edges)
    df_links = restructure_df_edges(df_orig)
    df_links = append_trackid_to_links(df_cells, df_links)
    print('Done.')
    
    return df_cells, df_links


###################################

def remove_division_links(df_cells, df_links):
    
    print('Removing links for dividing cells...')
    
    # cells that appear twice as target in the links
    source_count = Counter(list(df_links.source_id.values))
    to_drop = [key for key, val in source_count.items() if val == 2]
    
    df_links = df_links[~df_links.source_id.isin(to_drop)]
    df_links = df_links.reset_index(drop=True)
    
    return df_links

def remove_big_jumps(df_links, thr=20):

    print('Removing links for jumps bigger than', thr)

    df_links = df_links[df_links.speed<thr]

    return df_links

def remove_unlinked_cells(df_cells, df_links):
    
    print('Removing disconnected cells (those that do not appear in links)...')

    linked_cells = list(df_links.source_id.values) + list(df_links.target_id.values)
    linked_cells = list(set(linked_cells))
    
    df = df_cells[df_cells.id.isin(linked_cells)]
    df = df.reset_index(drop=True)
    
    return df

###################################

def reconstruct_tracks(df_links):
    
    print('Reconstructing cell tracks...')
    
    links = df_links[['source_id', 'target_id']].to_numpy()
    
    source = list(df_links.source_id.values)
    target = list(df_links.target_id.values)

    start = list(set(source)-set(target))
    finish = list(set(target)-set(source))

    assert len(start)==len(finish)
    
    tracks = np.zeros((1,2))
    track_id = 0
    for s in tqdm(start):
        track = [s]
        while track[-1] not in finish:
            track.append(links[links[:,0]==track[-1]][0][1])
        track = np.array(track)
        tracks = np.concatenate((tracks,np.array([
                            np.array([track_id for i in track]),
                            track]).T), axis=0)

        track_id += 1
        
    tracks = tracks[1:,:]
        
    return tracks.astype(int)

def append_track_id(df_cells, tracks):
    
    print('Creating cell_id column in cells dataframe...')
    
    tracks = tracks[tracks[:,1].argsort()].astype(int)
    df_cells = df_cells.sort_values(by=['id'])
    
    assert all(tracks[:,1]==df_cells.id.values)
    
    df_cells['cell_id'] = tracks[:,0]
        
    return df_cells
    

def append_n_spots_track(df_cells):
    print('Compute track length...')
    
    counts = df_cells['cell_id'].value_counts()
    df_cells['n_spots'] = df_cells['cell_id'].map(counts)
    
    return df_cells

def append_speed_tracks(df_cells, df_links):
    
    print('Compute track speed...')
    
    df_cells['source_id'] = df_cells.id
    df_cells = df_links.merge(df_cells, how='outer', on=['source_id'])
    
    return df_cells

def cleanup_cells_df(df_cells):
    
    print('Cleanup')
    
    df_cells = df_cells.drop(['id','source_id','target_id','track_id_x','track_id_y','n_links','track_n_spots'],axis=1)
    df_cells = df_cells.sort_values(by=['cell_id','t'])
    df_cells = df_cells[['cell_id','n_spots','r','x','y','z','t','speed','mean-ch1','mean-ch2']]
    df_cells = df_cells.reset_index(drop=True)
    
    return df_cells

def set_channels_threshold(df_cells, 
                            lims_ch1=[120,1000],
                            lims_ch2=[200,2000]):

    cell_ids = list(set(df_cells.cell_id.values))
    cell_ids.sort()
    # print(len(cell_ids))

    cells_to_keep = []

    for cell_id in tqdm(cell_ids, total=len(cell_ids)):
        cell = df_cells[df_cells.cell_id==cell_id]
        # print(cell_id, np.max(cell['mean-ch1'].values))
        if all([
                np.min(cell['mean-ch1'])>lims_ch1[0],
                np.max(cell['mean-ch1'])<lims_ch1[1],
                np.min(cell['mean-ch2'])>lims_ch2[0],
                np.max(cell['mean-ch2'])<lims_ch2[1]
                ]):
            cells_to_keep.append(cell_id)
    df_cells = df_cells[df_cells.cell_id.isin(cells_to_keep)]
    df_cells = df_cells.reset_index(drop=True)
    # print(len(cells_to_keep))

    return df_cells

def compute_track_info(df_cells):

    print('Computing track info...')

    # compute track df
    cell_ids = list(set(df_cells.cell_id))

    mean_speed = []
    mean_ch1 = []
    mean_ch2 = []
    end2end = []
    cumdist = []
    track_len = []
    start_track = []
    end_track = []

    for cell_id in tqdm(cell_ids, total=len(cell_ids)):
        df_cell = df_cells[df_cells.cell_id==cell_id]
        df_cell = df_cell.sort_values(by=['t'])
        track_len.append(len(df_cell))
        start_track.append(np.min(df_cell.t))
        end_track.append(np.max(df_cell.t))
        
        pos = df_cell[['x','y','z']].to_numpy()
        
        mean_speed.append(np.mean(df_cell.speed))
        mean_ch1.append(np.mean(df_cell['mean-ch1'].values))
        mean_ch2.append(np.mean(df_cell['mean-ch2'].values))
        end2end.append(np.linalg.norm(pos[0]-pos[-1]))
        cumdist.append(np.sum(np.sqrt(np.sum(np.diff(pos,axis=0)**2,axis=1))))
        

    df_tracks = pd.DataFrame({
        'cell_id':cell_ids,
        'len':track_len,
        'start_t':start_track,
        'end_t':end_track,
        'speed':mean_speed,
        'ch1':mean_ch1,
        'ch2':mean_ch2,
        'end2end':end2end,
        'cumdist':cumdist    
    })

    return df_tracks

###############################################################################

if __name__=='__main__':
    df_cells, df_links = process_csv(vertices = 'FeatureAndTagTable-vertices.csv',
                edges = 'FeatureAndTagTable-edges.csv')
    
    df_links = remove_division_links(df_cells, df_links)
    df_cells = remove_unlinked_cells(df_cells, df_links)
    
    tracks = reconstruct_tracks(df_links)
    df_cells = append_track_id(df_cells, tracks)
    df_cells = append_n_spots_track(df_cells)
    df_cells = append_speed_tracks(df_cells, df_links)
    df_cells = cleanup_cells_df(df_cells)
    
    print('Done.')
    

            



