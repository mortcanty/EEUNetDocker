#!/usr/bin/env python
#******************************************************************************
#******************************************************************************
#  Name:     s2cloudless.py (colab version)
#  Purpose:  Create a cloud-free image from GEE
#  Taken from Earthengine Community Tutorial by Justin Braaten
#  Usage (from command line):             
#    python s2cloudless.py  [options] areaOfInterest fileNamePrefix

import ee
import sys, getopt

ee.Initialize()

CLOUD_FILTER = 60
CLD_PRB_THRESH = 40
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 100

def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))
    
def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)    

def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)

    
def main(): 

    usage = '''            
Usage: 
--------------------------------------

Create a cloud-free Sentinel-2 surface reflectance image from GEE

python %s [OPTIONS] lonlat fnprefix

Options:
  -h            this help
  -s  <string>  start date (default '2020-06-01')
  -e  <string>  end date (default '2020-09-01')
  
coords:     region of interest interest (list)  
fnprefix:   filename (no extension) for export to Drive as gee/<fileNamePrefix>
  
  -------------------------------------'''%sys.argv[0]            
                    
    options,args = getopt.getopt(sys.argv[1:],'hnm:t:d:s:v:')

    start_date = '2020-06-01'
    end_date = '2020-09-01'
    for option, value in options: 
        if option == '-h':
            print(usage)
            return 
        elif option == '-s':
            start_date = value
        elif option == '-e':
            end_date = value
    if len(args) != 2:
        print( 'Incorrect number of arguments' )
        print(usage)
        sys.exit(1)  
        
    coords = eval(sys.argv[1])
    aoi = ee.Geometry.Polygon(coords)
    fnprefix = sys.argv[2]
    
    print('---------------------------------------------------')    
    print('Creating cloudless S2 composite for period %s to %s'%(start_date,end_date))
    print('---------------------------------------------------')
    
    s2_sr_cld_col = get_s2_sr_cld_col(aoi, start_date, end_date)       
    
    s2_sr_median = (s2_sr_cld_col.map(add_cld_shdw_mask)
                             .map(apply_cld_shdw_mask)
                             .median()) 
                             
    crs = ee.ImageCollection('COPERNICUS/S2_SR') \
                      .filterBounds(aoi) \
                      .filterDate(ee.Date(start_date),ee.Date(end_date)) \
                      .first() \
                      .select(0).projection().crs()
                      
    gdexport = ee.batch.Export.image.toDrive(s2_sr_median.select(1,2,3).clip(aoi),
                                              description='driveExportTask_s2', 
                                              folder = 'gee',
                                              fileNamePrefix = fnprefix,
                                              crs = crs,
                                              scale = 10,
                                              maxPixels = 1e11)
    gdexport.start()

if __name__ == '__main__':
    main()    
    
    