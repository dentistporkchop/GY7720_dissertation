import os
import joblib
import numpy as np
import rasterio as rio
from rasterio.plot import reshape_as_image
import sys
from os.path import join
from datetime import datetime
import re

scenes_folder = sys.argv[1]
os.chdir(scenes_folder)
scenes = os.listdir(scenes_folder)

scenes.sort()

pkl_dir = "/home/b/bac14/Documents/dissertation/data/pkl_files"

for scene in scenes:

    raster = scene + re.sub(r'^.*?_', '/', scene) + ".tif"
    #"/home/b/bac14/Documents/dissertation/data/validation/scenes/Ravine_S2A_20191216_S2B_20200130_T53HPA_R102/S2A_20191216_S2B_20200130_T53HPA_R102.tif"
    src = rio.open(raster, 'r')
    img = src.read()
    orig_shape = (img.shape[1], img.shape[2])

    scene_name = raster.rsplit("/",1)[-1][:-4]

    pre_date = datetime.strptime(scene_name.rsplit('_')[1], '%Y%m%d')
    post_date = datetime.strptime(scene_name.rsplit('_')[3], '%Y%m%d')
    interval_days = (post_date - pre_date).days
    pre_month = pre_date.month

    pre_month_array = np.full(orig_shape, pre_month)
    interval_days_array = np.full(orig_shape, interval_days)

    pre_b4 = img[0]
    pre_b8 = img[1]
    pre_b12 = img[2]
    post_b4 = img[3]
    post_b8 = img[4]
    post_b12 = img[5]
    
    pre_NBR = (pre_b8 - pre_b12)/(pre_b8 + pre_b12)
    post_NBR = (post_b8 - post_b12)/(post_b8 + post_b12)
    dNBR = post_NBR - pre_NBR

    for band in [pre_NBR, post_NBR, dNBR]:
        np.nan_to_num(band, copy=False, nan=-9999)

    new_columns = np.array([pre_b4, pre_b8, pre_b12, post_b4, post_b8, post_b12, pre_NBR, post_NBR, dNBR, pre_month_array, interval_days_array])

    n_bands = new_columns.shape[0]
    
    print("Number of bands for classification: ", n_bands)
    print("new_columns shape: ", new_columns.shape)
    
    print("Pre-reshape shape: ", new_columns.shape)
    new_columns = reshape_as_image(new_columns)
    print("Reshaped shape   : ", new_columns.shape)
    
    new_shape = (new_columns.shape[0] * new_columns.shape[1], 
                 new_columns.shape[2]) 
    
    print("img[:, :, :n].shape = ", new_columns[:, :, :n_bands].shape)
    print("new_shape = ", new_shape)
    
    img_as_array = new_columns[:, :, :n_bands].reshape(new_shape)
    
    print(img_as_array)
    
    if str(sys.argv[2]) == "pca":
        scaler = joblib.load(join(pkl_dir, "MinMaxScaler.pkl"))
        pca = joblib.load(join(pkl_dir, "pca.pkl"))
        rf = joblib.load(join(pkl_dir, "rf_pca.pkl"))
        
        img_as_array = scaler.transform(img_as_array)
        X = pca.transform(img_as_array)
        val_type = "_pca_classification.tif"
    else:
        rf = joblib.load(join(pkl_dir, "rf_baseline.pkl"))
        
        X = img_as_array
        val_type = "_bl_classification.tif"
    
    val_rf = rf.predict(X)
    val_rf_proba = rf.predict_proba(X)
    
    proba_1 = val_rf_proba[:, 0]
    proba_2 = val_rf_proba[:, 1]
    proba_3 = val_rf_proba[:, 2]
    
    val_array = np.reshape(val_rf, orig_shape)
    p1_array = np.reshape(proba_1, orig_shape)
    p2_array = np.reshape(proba_2, orig_shape)
    p3_array = np.reshape(proba_3, orig_shape)
    
    # Register GDAL format drivers and configuration options with a
    # context manager.
    
    tif_path = raster.rsplit("/",1)[0]
    file_name = scene_name + val_type
    
    
    with rio.Env():
    
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = src.profile
    
        # And then change the band count to 4, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(
            dtype=rio.float32,
            count=4,
            compress='lzw')
    
    
    
        with rio.open(join(tif_path, file_name), 
                      'w', **profile) as dst:
            dst.write(val_array.astype(rio.float32), 1)
            dst.write(p1_array.astype(rio.float32), 2)
            dst.write(p2_array.astype(rio.float32), 3)
            dst.write(p3_array.astype(rio.float32), 4)
    
    # At the end of the ``with rasterio.Env()`` block, context
    # manager exits and all drivers are de-registered.