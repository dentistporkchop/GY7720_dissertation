# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Load necessary libraries
import geopandas as gpd
import os
from osgeo import gdal
import rasterio as rio
import rasterio.plot
from rasterio.mask import mask
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import sys
import re
from os.path import join

scenes_path = "/home/b/bac14/Documents/dissertation/data/validation/scenes"#sys.argv[1]
os.chdir(scenes_path)

aux_path = join(scenes_path.rsplit('/', 1)[0], "acc_assess_output", "auxiliary_files")
if not os.path.exists(aux_path):
    os.mkdir(aux_path)

scenes = os.listdir(scenes_path)

scenes.sort()

scenes_each = [scenes[1]]

for scene in scenes_each:
    
    single_scene_path = join(scenes_path, scene)
    
    #scene = scene.rsplit('/', 1)[-1]
    scene_area = scene.rsplit('_')[0]
    
    ba_tif = scene + re.sub(r'^.*?_', '/', scene) + "_pca_classification.tif"
    ref_shp = join(scenes_path.rsplit('/', 1)[0], 
                   "reference_shapefiles", 
                   scene_area + "_burn_ref.shp")
    
    # Define path of shapefile and gdf
    #path = "C:\\Users\\benco\\OneDrive\\Documents\\Uni\\AI4EO\\Accuracy_Assessment\\sites\\Australia\\Upper-Murray-2_S2B_20191119_S2A_20191214_T55HEV_R073\\burn_ref.shp"
    
    ref_gdf = gpd.read_file(ref_shp)
    
    # Read raster rf classification with rasterio
    ba_raster = rio.open(ba_tif)
    
    # Match shapefile crs with crs of tile
    raster_proj = ba_raster.crs
    ref_gdf = ref_gdf.to_crs(raster_proj)
    
    # Create RF mask
    out_image, out_transform = rio.mask.mask(ba_raster, ref_gdf['geometry'], crop=False)  
    out_image = out_image.astype('int16')
    out_meta = ba_raster.meta
    
    # UPDATE METAFILE WITH CLIPPED IMAGE    
    out_meta.update({"driver": "GTiff",
                     "count": 1,
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    
    class_array = ba_raster.read(1).astype('int16')
    ref_array = out_image[0][np.newaxis]
    ref_array[ref_array==-9999] = 0
    ref_array.astype('int16')
    
    def compare_class(a, b):
        if a==0:
            if b==1:
                return 4
            elif b==2:
                return 5
            elif b==3:
                return 2
        else:
            if b==1:
                return 1
            elif b==2:
                return 5
            elif b==3:
                return 3
    
    vfunc = np.vectorize(compare_class)
    
    comparison_array = vfunc(ref_array, class_array)
    
    # WRITE CLIPPED IMAGE TO FILE
    
    var1 = scene + '_acc_assessment_var1'
    var2 = scene + '_acc_assessment_var2'
    
    
    with rasterio.open(join(single_scene_path, var2 + '.tif'), "w", **out_meta) as dest:
        dest.write(comparison_array)#[np.newaxis]) # need to add dimension for some reason
        
    var2_raster = rio.open(join(single_scene_path, var2 + '.tif'))
    var2_band1 = var2_raster.read(1)
    
    unique, counts = np.unique(var2_band1, return_counts=True)
    cat_count = dict(zip(unique, counts))
    
    print(scene + ":\n", cat_count)
    
    # Set class colours for plotting image (#d3d3d3 is a light grey)
    levels = [0.9, 1.9, 2.9, 3.9, 4.9]
    colours = ["#8b0000", "#FFFFE0", "#00FFFF", "#ffb780", "black"]
    
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colours, extend = 'max')
    
    fig, ax = plt.subplots(figsize=(15,15))
    
    class_plot = ax.imshow(var2_band1,
                           cmap=cmap,
                           norm=norm, 
                           interpolation='none')
    
    #ax.set_title(folder_path.split("/")[-1] + " accuracy assessment (variant 2)")
    ax.set_title(scene + " accuracy assessment (variant 2)")
    
    legend_labels = {"#8b0000": "Agree Burned",
                     "#FFFFE0": "Agree Unburned",
                     "#00FFFF": "Reference Only Detected",
                     "#ffb780": "AIBAM Only Detected",
                     "black": "Not Known/No Data"}
    
    patches = [Patch(color=color, label=label)
               for color, label in legend_labels.items()]
    
    ax.legend(handles=patches,
             facecolor="white",
             bbox_to_anchor=(1,1),
             loc='upper left')
    
    def myround50(x, base=100):
        return base * round(x/base)
    
    scale_size_pixels = myround50(var2_band1.shape[1]/5)
    scale_size_km = int(scale_size_pixels/100)
    
    scalebar = AnchoredSizeBar(ax.transData,
                              var2_band1.shape[1]/5, #2000,
                              str(scale_size_km) + ' km',
                              'lower left',
                              pad=0.5,
                              borderpad=1,
                              sep=5,
                              color='black',
                              frameon=True,
                              size_vertical=2)
    ax.add_artist(scalebar)
    
    # ax.set_xlim(6000,10000)
    # ax.set_ylim(2600,1000)
    ax.set_axis_off()
    
    fig.savefig(join(single_scene_path, var2 + '.png'), bbox_inches='tight',pad_inches = 0)
    
    '''
    CREATING VARIANT 1
    from variant 2
    '''
    
    minx, miny, maxx, maxy = ref_gdf.geometry.total_bounds
    premask_tif = join(aux_path, scene + "_premask.tif")
    gdal.Warp(destNameOrDestDS=premask_tif, srcDSOrSrcDSTab= join(single_scene_path, var2 + '.tif'), outputBounds=(minx, miny, maxx, maxy))
    
    see = rio.open(premask_tif)
    see = see.read(1, masked=True)
    
    ### try this out
    with rasterio.open(premask_tif) as src:  
        bands = src.read(masked=True).astype('int16')
        bands.fill_value = -9999
        
        kwargs = src.meta.copy()
        kwargs.update({'nodata': -9999})
        src.close()
    
    copy_premask_tif = join(aux_path, scene + "_copypremask.tif")
    
    with rasterio.open(copy_premask_tif, 'w', **kwargs) as dst:
        dst.write(bands)
        
    var1_premask = rio.open(copy_premask_tif)
    out_image, out_transform = rio.mask.mask(var1_premask, ref_gdf['geometry'], crop=True)  
    out_image = out_image.astype('int16')
    out_meta = var1_premask.meta
    
    # UPDATE METAFILE WITH CLIPPED IMAGE    
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    
    # WRITE CLIPPED IMAGE TO FILE
    with rasterio.open(join(single_scene_path, var1 + '.tif'), "w", **out_meta) as dest:
        dest.write(out_image)
    '''
    PLOTTING VARIANT 1
    '''
    
    var1_raster = rio.open(join(single_scene_path, var1 + '.tif'))
    var1_band1 = var1_raster.read(1)
    
    # Set class colours for plotting image (#d3d3d3 is a light grey)
    levels = [-9999.1, -0.1, 0.9, 2.9, 4.9]
    colours = ["white", "black", "#8b0000", "#00FFFF", "black"]
    
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colours, extend = 'max')
    
    fig, ax = plt.subplots(figsize=(15,15))
    
    class_plot = ax.imshow(var1_band1,
                           cmap=cmap,
                           norm=norm, 
                           interpolation='none')
    
    #ax.set_title(folder_path.split("/")[-1] + " accuracy assessment (variant 1)")
    ax.set_title(scene + " accuracy assessment (variant 1)")
    
    legend_labels = {"#8b0000": "Agree Burned",
                     "#00FFFF": "Reference Only Detected",
                     "black": "Not Known/No Data"}
    
    patches = [Patch(color=color, label=label)
               for color, label in legend_labels.items()]
    
    ax.legend(handles=patches,
             facecolor="white",
             bbox_to_anchor=(1,1),
             loc='upper left')
    
    scale_size_pixels = myround50(var1_band1.shape[1]/5)
    scale_size_km = int(scale_size_pixels/100)
    
    scalebar = AnchoredSizeBar(ax.transData,
                              var2_band1.shape[1]/5, #2000,
                              str(scale_size_km) + ' km',
                              'lower left',
                              pad=0.5,
                              borderpad=1,
                              sep=5,
                              color='black',
                              frameon=True,
                              size_vertical=1)
    ax.add_artist(scalebar)
    
    ax.set_axis_off()
    
    fig.savefig(join(single_scene_path, var1 + '.png'), bbox_inches='tight',pad_inches = 0)
    
    var1_raster.close()
    var1_premask.close()
    var2_raster.close()
    #see.close()
