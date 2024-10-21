import numpy as np
import sys
import tifffile
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
from dynamics import gen_pose_target
from transforms import make_tiles, normalize99


def center_crop(img, dim):
    """Returns center cropped image

    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]
    #process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

def get_tile_dims(img_path, technology, d, border=0):
    image = membrane = dapi = None
    if technology == "CosMx":
        image = tifffile.imread(img_path)
        membrane = np.sum(image[0:4, :, :], axis=0)
        dapi = image[4, :, :]
        image = normalize99(np.stack([membrane, dapi]))
    
    if technology == "Xenium":
        # reader = OMETIFFReader(fpath=img_path)
        # image, metadata, xml_metadata = reader.read()
        
        # image = loaded
        image = tifffile.imread(img_path)
        image = image[:, 5000:10000, 30000:35000]
        membrane = np.sum(image[1:, :, :], axis=0)
        dapi = image[0, :, :]
        image = normalize99(np.stack([membrane, dapi]))

    if technology == "Custom":
        # image = tifffile.imread(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, 0)
        image = normalize99(image)

    sh1 = image.shape[1]
    sh2 = image.shape[2]

    # remove border
    image = image[:, border:sh1-border, border:sh2-border]

    # make tiles
    tiles, ysub, xsub, Ly, Lx = make_tiles(image, bsize=d)

    return (tiles.shape[0], tiles.shape[1])

def process_transcripts(tx_path, fov: int, z: int, save_path):
    """CosMx Transcripts"""
    tx = pd.read_csv(tx_path)
    locations = tx.loc[tx['fov'] == fov]
    locations = locations.loc[locations['z'] == z]
    locations = locations.drop(columns=['fov', 'cell_ID', 'x_global_px', 'y_global_px', 'z', 'target', 'CellComp'])
    locations.to_csv(save_path)

def process_parquet(transcripts_path, x_min, x_max, y_min, y_max, save_path, pixel_size=0.2125):
    """Xenium Transcripts
    
    see image scale factors for pixel size: https://kb.10xgenomics.com/hc/en-us/articles/11636252598925-What-are-the-Xenium-image-scale-factors
    """
    # x_min, x_max, y_min, y_max = [r * pixel_size for r in [x_min, x_max, y_min, y_max]]
    transcripts = pd.read_parquet(transcripts_path, engine='pyarrow')

    transcripts['x_location'] = transcripts['x_location'] / pixel_size
    transcripts['y_location'] = transcripts['y_location'] / pixel_size

    transcripts = transcripts.loc[(transcripts['x_location'] > x_min) & (transcripts['x_location'] < x_max)]
    print(len(transcripts.values))
    transcripts = transcripts.loc[(transcripts['y_location'] > y_min) & (transcripts['y_location'] < y_max)]

    transcripts = transcripts.drop(columns=[
        'transcript_id', 'cell_id', 'overlaps_nucleus', 'feature_name', 'z_location', 'qv', 'fov_name', 'nucleus_distance', 'codeword_index', 'codeword_category', 'is_gene'])
    transcripts.to_csv(save_path)

def process_pci(transcripts_path, save_path):
    transcripts = pd.read_csv(transcripts_path)
    transcripts = transcripts.drop(columns=['gene'])
    transcripts.to_csv(save_path)

def process_morphology(img_path, technology, d, j, i, border=0, loaded=None):

    #load

    image = membrane = dapi = None
    if technology == "CosMx":
        image = tifffile.imread(img_path)
        membrane = np.sum(image[0:4, :, :], axis=0)
        dapi = image[4, :, :]
        image = normalize99(np.stack([membrane, dapi]))
    
    if technology == "Xenium":
        # reader = OMETIFFReader(fpath=img_path)
        # image, metadata, xml_metadata = reader.read()
        
        image = loaded
        #image = tifffile.imread(img_path)
        image = image[:, 5000:10000, 30000:35000]
        membrane = np.sum(image[1:, :, :], axis=0)
        dapi = image[0, :, :]
        image = normalize99(np.stack([membrane, dapi]))

    if technology == "Custom":
        # image = tifffile.imread(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, 0)
        image = normalize99(image)

    sh1 = image.shape[1]
    sh2 = image.shape[2]

    # remove border
    image = image[:, border:sh1-border, border:sh2-border]

    # make tiles
    tiles, ysub, xsub, Ly, Lx = make_tiles(image, bsize=d)

    return tiles[j, i, :, :, :], (sh1, sh2), ysub[j * d + i], xsub[j * d + i]


def process_spots(rna, image_shape, ysub, xsub, j, i, border=0, x_min=0, x_max=0, y_min=0, y_max=0):
    spots = []
    yrange = [y + y_min for y in ysub]
    xrange = [x + x_min for x in xsub]
    
    for spot in rna:
        loc = []
        if xrange[0] < spot[0] < xrange[1]:
            loc.append(spot[0])
        if yrange[0] < spot[1] < yrange[1]:
            loc.append(spot[1])
        if len(loc) == 2:
            spots.append([loc[0] - xrange[0], loc[1] - yrange[0]])

    return spots
