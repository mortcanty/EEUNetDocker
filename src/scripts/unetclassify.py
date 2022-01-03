#!/usr/bin/env python
#******************************************************************************
#  Name:     unetcnnclass.py
#  Purpose:  Use FCN to find buildings in a hi-res rgbn image
#  Usage (from command line):             
#    python unetcnnclass.py  [options] filename
#
#  Copyright (c) 2021 Mort Canty

import sys, os, time, getopt
import tensorflow.keras as keras
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32

def bytestretch(arr,rng=None):
#  byte stretch image numpy array
    shp = arr.shape
    arr = arr.ravel()
    if rng is None:
        rng = [np.min(arr),np.max(arr)]
    tmp =  (arr-rng[0])*255.0/(rng[1]-rng[0])
    tmp = np.where(tmp<0,0,tmp)  
    tmp = np.where(tmp>255,255,tmp) 
    return np.asarray(np.reshape(tmp,shp),np.float32)
    
def lin2pcstr(x):
#  2% linear stretch
    x = bytestretch(x)
    hist,bin_edges = np.histogram(x,256,(0,256))
    cdf = hist.cumsum()
    lower = 0
    i = 0
    while cdf[i] < 0.02*cdf[-1]:
        lower += 1
        i += 1
    upper = 255
    i = 255
    while (cdf[i] > 0.98*cdf[-1]) and (upper>100):
        upper -= 1
        i -= 1
    fp = (bin_edges-lower)*255/(upper-lower)
    fp = np.where(bin_edges<=lower,0,fp)
    fp = np.where(bin_edges>=upper,255,fp)
    return np.interp(x,bin_edges,fp)     
    
def main(): 
    global model
    
    usage = '''            
Usage: 
--------------------------------------

Deep learning semantic classification of hi-res images
for building identification

python %s [OPTIONS] filename

Options:
  -h            this help
  -m  <string>  path to stored model        
  -d  <list>    spatial subset [x,y,width,height]
  -t  <float>   probability threshold (default 0.4)
  
Classes:                   
                    0 'All background',
                    1 'Buildings',
  
Assumes rgb image bands only 
  
  -------------------------------------'''%sys.argv[0]            
                    
    options,args = getopt.getopt(sys.argv[1:],'hm:d:t:')
    model_path = '/media/mort/Crucial/imagery/Inria/AerialImageDataset/train/unet_inria_modelx.h5'
    dims = None
    threshold = 0.4
    for option, value in options: 
        if option == '-h':
            print(usage)
            return 
        elif option == '-m':
            model_path = value
        elif option == '-d':
            dims = eval(value)  
        elif option == '-t':
            threshold = eval(value)  
    if len(args) != 1:
        print( 'Incorrect number of arguments' )
        print(usage)
        sys.exit(1)                
        
    gdal.AllRegister()
    
    try:   
# #      blob detector
#         params = cv2.SimpleBlobDetector_Params() 
#         params.minThreshold = 100
#         params.maxThreshold = 256
#         params.filterByCircularity = False
#         params.filterByInertia = False
#         params.filterByConvexity = False
#         params.filterByColor=False
#         params.filterByArea = True
#         params.minArea = 20
#         params.maxArea = 1e5
#         detector = cv2.SimpleBlobDetector_create(params) 
#         detector.empty()
#      load the trained CNN model     
        model = keras.models.load_model(model_path)   
#      read in rgb image           
        infile = args[0] 
        path = os.path.dirname(infile)
        basename = os.path.basename(infile)
        root, ext = os.path.splitext(basename)
        outfile = path+'/'+root+'_cnn'+ext         
        start = time.time()           
        inDataset = gdal.Open(infile, GA_ReadOnly)
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize     
        if dims:
            x0,y0,cols,rows = dims
        else:
            x0 = 0
            y0 = 0                   
        G = np.zeros((rows,cols,4))                               
        for b in range(4):
            band = inDataset.GetRasterBand(b+1)
            G[:,:,b] = lin2pcstr(band.ReadAsArray(x0,y0,cols,rows))   
#      classify in patches    
        xpatches = cols//512
        ypatches = rows//512       
        result = np.zeros((ypatches*512,xpatches*512))       
        for iy in range(ypatches):
            for ix in range(xpatches):
                patch = G[iy*512:(iy+1)*512,ix*512:(ix+1)*512,:]
                patch = patch.reshape(1,512,512,4)
                res = np.reshape(model.predict(patch)[0],(512,512))
                result[iy*512:(iy+1)*512,ix*512:(ix+1)*512] = res
        result = np.where(result<threshold,0,255) 
        im = result.astype(np.uint8)
#        keypoints = detector.detect(im)
#        print('Number of structures: %i'%len(keypoints))
    except Exception as e:
        print('Error: %s'%e)             
#  write to disk    
    driver =  gdal.GetDriverByName('GTiff')
    outDataset = driver.Create(outfile,xpatches*512,ypatches*512,1,GDT_Float32)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)               
    outBand = outDataset.GetRasterBand(1)  
    outBand.WriteArray(result[:,:],0,0) 
    outBand.FlushCache() 
    inDataset = None
    outDataset = None
    print('Map written to: %s'%outfile)         
    print('Elapsed time: %s'%str(time.time()-start))          
    
if __name__ == '__main__':
    main()    
    
    