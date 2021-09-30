import sys
from osgeo import gdal, ogr
import os
from os.path import join
import re
import numpy
import rasterio as rio
import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import zipfile
import shutil

#-----
# From pyeo
def convert_shapefile2array(trainingfnm,img_ds_str,nodata_value,options=["ATTRIBUTE=id"]):

    img_ds = gdal.Open(img_ds_str)

    trng_ds=ogr.Open(trainingfnm)
    trng_layer=trng_ds.GetLayer()
    trng_rds = gdal.GetDriverByName("MEM").Create("",img_ds.RasterXSize,img_ds.RasterYSize,1,gdal.GDT_Int16)
    trng_rds.SetGeoTransform(img_ds.GetGeoTransform())
    trng_rds.SetProjection(img_ds.GetProjection())
    band = trng_rds.GetRasterBand(1)
    band.SetNoDataValue(-1)
    tra = numpy.zeros((img_ds.RasterYSize,img_ds.RasterXSize), dtype = numpy.int32)
    tra[:]=nodata_value
    trng_rds.GetRasterBand(1).WriteArray(tra)
    gdal.RasterizeLayer(trng_rds, [1], trng_layer, burn_values=[nodata_value],options=options)
    tra=trng_rds.GetRasterBand(1).ReadAsArray()
    trng_ds,trng_rds=None,None
    return tra

# Set working directory
wd = sys.argv[1]
os.chdir(wd)

# Create list of Sentinel tiles in working directory (files should have training data
# and at least one tif of the scene (need the dimensions))
scenes = [f for f in os.listdir(wd) if re.search(r"\AS2[AB]_2", f) and zipfile.is_zipfile(f)]

#-----

#master_df = pd.DataFrame(columns = ['scene',
#                                    'pre_date',
#                                    'post_date',
#                                    'interval_days',
#                                    'row',
#                                    'col',
#                                    'pre_b4',
#                                    'pre_b8',
#                                   'pre_b12',
#                                    'post_b4',
#                                    'post_b8',
#                                    'post_b12',
#                                    'pre_NBR',
#                                    'post_NBR',
#                                    'dNBR',
#                                    'burn_class'])
#-----

for scene in scenes:
    
    with zipfile.ZipFile(scene, 'r') as zip:
        zip.extractall()
    
    folder = scene[:-4]
    img_ds = join(folder, [f for f in os.listdir(folder) if re.search(r"[0-9].tif\Z", f)][0])
    shp = join(folder, 'training/training.shp')

    burn_class = convert_shapefile2array(shp,img_ds,-9999)
    
    with rio.open(img_ds) as src:
        pre_4 = src.read(1)
        pre_8 = src.read(2)
        pre_12 = src.read(3)
        po_4 = src.read(4)
        po_8 = src.read(5)
        po_12 = src.read(6)
        shape = src.shape
        src.close()
        
    try:
        shutil.rmtree(folder)
    except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    
    pre_NBR = (pre_8 - pre_12)/(pre_8 + pre_12)
    post_NBR = (po_8 - po_12)/(po_8 + po_12)
    dNBR = post_NBR - pre_NBR
    
    training_arrays = {
        'pre_b4': np.ravel(pre_4),
        'pre_b8': np.ravel(pre_8),
        'pre_b12': np.ravel(pre_12),
        'post_b4': np.ravel(po_4),
        'post_b8': np.ravel(po_8),
        'post_b12': np.ravel(po_12),
        'pre_NBR': np.ravel(pre_NBR),
        'post_NBR': np.ravel(post_NBR),
        'dNBR': np.ravel(dNBR),
        'burn_class': np.ravel(burn_class)
    }

    df = pd.DataFrame.from_dict(training_arrays)
    
    # Preserve positional attributes before removing no data vals
    df['col'] = list(range(1,shape[1]+1)) * shape[0]
    lst = list(range(1,shape[0]+1))
    df['row'] = list(itertools.chain.from_iterable(itertools.repeat(x, shape[1]) for x in lst))
    
    df = df[df.burn_class != -9999]
    df['scene'] = folder
    df['pre_date'] = datetime.strptime(folder.rsplit('_')[1], '%Y%m%d')
    df['post_date'] = datetime.strptime(folder.rsplit('_')[3], '%Y%m%d')
    df['interval_days'] = (df['post_date'] - df['pre_date']).apply(lambda x: x.days)
    df['pre_month'] = df['pre_date'].apply(lambda x: x.month)
    
    master_df = pd.concat([master_df, df]).reset_index(drop=True)
    
master_df.to_csv("/home/b/bac14/Documents/dissertation/data/aggregated_training_data.csv", index=False)