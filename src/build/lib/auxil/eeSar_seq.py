'''
Created on 21.06.2018

@author: mort
'''
import ee, time, warnings
import ipywidgets as widgets
import numpy as np
from IPython.display import display
from ipyleaflet import (Map,DrawControl,TileLayer,basemaps,basemap_to_tiles,SplitMapControl)
from auxil.eeWishart import omnibus

ee.Initialize()

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

poly = ee.Geometry.Polygon([[6.30154, 50.948329], [6.293307, 50.877329], 
                            [6.427091, 50.875595], [6.417486, 50.947464], 
                            [6.30154, 50.948329]])

center = list(reversed(poly.centroid().coordinates().getInfo()))

def get_incidence_angle(image):
    ''' grab the mean incidence angle '''
    result = ee.Image(image).select('angle') \
           .reduceRegion(ee.Reducer.mean(),geometry=poly,maxPixels=1e9) \
           .get('angle') \
           .getInfo()
    if result is not None:
        return round(result,2)
    else:
#      incomplete overlap        
        return round(ee.Image(image).select('angle') \
           .reduceRegion(ee.Reducer.mean(),maxPixels=1e9) \
           .get('angle') \
           .getInfo(),2)

def get_vvvh(image):
    ''' get 'VV' and 'VH' bands from sentinel-1 imageCollection '''
    return image.select('VV','VH')

def get_image(current,image):
    ''' accumulate a single image from a collection of images '''
    return ee.Image.cat(ee.Image(image),current)    
    
def clipList(current,prev):
    ''' clip a list of images '''
    imlist = ee.List(ee.Dictionary(prev).get('imlist'))
    poly = ee.Dictionary(prev).get('poly')    
    imlist = imlist.add(ee.Image(current).clip(poly))
    return ee.Dictionary({'imlist':imlist,'poly':poly})

def handle_draw(self, action, geo_json):
    global poly
    if action == 'created':
        coords =  geo_json['geometry']['coordinates']
        poly = ee.Geometry.Polygon(coords)
        w_preview.disabled = True
        w_export.disabled = True
        
dc = DrawControl(polyline={},circle={})
dc.on_draw(handle_draw)

def GetTileLayerUrl(ee_image_object):
    map_id = ee.Image(ee_image_object).getMapId()
    tile_url_template = "https://earthengine.googleapis.com/map/{mapid}/{{z}}/{{x}}/{{y}}?token={token}"
    return tile_url_template.format(**map_id)

w_orbitpass = widgets.RadioButtons(
    options=['ASCENDING','DESCENDING'],
     value='ASCENDING',
    description='Orbit pass:',
    disabled=False
)
w_changemap = widgets.RadioButtons(
    options=['First','Last','Frequency'],
     value='First',
    description='Change map:',
    disabled=False
)
w_platform = widgets.RadioButtons(
    options=['Both','A','B'],
     value='Both',
    description='Platform:',
    disabled=False
)
w_relativeorbitnumber = widgets.IntText(
    value=0,
    description='Rel orbit:',
    disabled=False
)
w_exportname = widgets.Text(
    value='users/<username>/<path>',
    placeholder=' ',
    disabled=False
)
w_startdate = widgets.Text(
    value='2017-04-01',
    placeholder=' ',
    description='Start date:',
    disabled=False
)
w_enddate = widgets.Text(
    value='2017-10-01',
    placeholder=' ',
    description='End date:',
    disabled=False
)
w_median = widgets.Checkbox(
    value=True,
    description='5x5 Median filter',
    disabled=False
)
w_significance = widgets.BoundedFloatText(
    value=0.0001,
    min=0,
    max=0.05,
    step=0.0001,
    description='Significance:',
    disabled=False
)
w_opacity = widgets.BoundedFloatText(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.1,
    description='Opacity:',
    disabled=False
)

w_text = widgets.Textarea(
    value = 'Algorithm output',
    rows = 4,
    disabled = False
)

w_run = widgets.Button(description="Run")
w_preview = widgets.Button(description="Preview",disabled=True)
w_export = widgets.Button(description='Export to assets',disabled=True)
w_dates = widgets.HBox([w_relativeorbitnumber,w_startdate,w_enddate])
w_orbit = widgets.HBox([w_orbitpass,w_platform,w_changemap,w_opacity])
w_exp = widgets.HBox([w_export,w_exportname])
w_signif = widgets.HBox([w_significance,w_median])
w_rse = widgets.HBox([w_run,w_preview,w_exp])

box = widgets.VBox([w_text,w_dates,w_orbit,w_signif,w_rse])

def on_widget_change(b):
    w_preview.disabled = True
    w_export.disabled = True

w_orbitpass.observe(on_widget_change,names='value')
w_platform.observe(on_widget_change,names='value')
w_relativeorbitnumber.observe(on_widget_change,names='value')
w_startdate.observe(on_widget_change,names='value')
w_enddate.observe(on_widget_change,names='value')
w_median.observe(on_widget_change,names='value')
w_significance.observe(on_widget_change,names='value')

def on_run_button_clicked(b):
    global result,m,collection,count,timestamplist1, \
           w_startdate,w_enddate,w_orbitpass,w_changemap, \
           w_relativeorbitnumber,w_significance,w_median, \
           collectionmean
    try:
        w_text.value = 'running...'
        collection = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') \
                  .filterBounds(poly) \
                  .filterDate(ee.Date(w_startdate.value), ee.Date(w_enddate.value)) \
                  .filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV','VH'])) \
                  .filter(ee.Filter.eq('resolution_meters', 10)) \
                  .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                  .filter(ee.Filter.eq('orbitProperties_pass', w_orbitpass.value))   
        if w_relativeorbitnumber.value > 0:
            collection = collection.filter(ee.Filter.eq('relativeOrbitNumber_start', int(w_relativeorbitnumber.value)))   
        if w_platform.value != 'Both':
            collection = collection.filter(ee.Filter.eq('platform_number', w_platform.value))         
        collection = collection.sort('system:time_start') 

        acquisition_times = ee.List(collection.aggregate_array('system:time_start')).getInfo()
        count = len(acquisition_times) 
        if count<2:
            raise ValueError('Less than 2 images found')
        timestamplist = []
        for timestamp in acquisition_times:
            tmp = time.gmtime(int(timestamp)/1000)
            timestamplist.append(time.strftime('%x', tmp))  
#      make timestamps in YYYYMMDD format            
        timestamplist = [x.replace('/','') for x in timestamplist]  
        timestamplist = ['T20'+x[4:]+x[0:4] for x in timestamplist]
#      in case of duplicates add running integer
        timestamplist1 = [timestamplist[i] + '_' + str(i+1) for i in range(len(timestamplist))]    
        relativeorbitnumbers = map(int,ee.List(collection.aggregate_array('relativeOrbitNumber_start')).getInfo())
        rons = list(set(relativeorbitnumbers))
        txt = 'Images found: %i, platform: %s \n'%(count,w_platform.value)
        txt += 'Acquisition dates: '+timestamplist[0]+'...'+timestamplist[-1]+'\n'
        txt += 'Relative orbit numbers: '+str(rons)+'\n'
        if len(rons)==1:
            txt += 'Mean incidence angle: %f'%get_incidence_angle(collection.first())
        else:
            txt += 'Mean incidence angle: (select a rel. orbit)'
        w_text.value = txt
        pcollection = collection.map(get_vvvh)
        pList = pcollection.toList(100)   
        first = ee.Dictionary({'imlist':ee.List([]),'poly':poly}) 
        imList = ee.Dictionary(pList.iterate(clipList,first)).get('imlist')
        result = ee.Dictionary(omnibus(imList,w_significance.value,w_median.value))
        w_preview.disabled = False
#      display mean of the full collection                
        collectionmean = collection.mean() \
                               .select(0) \
                               .clip(poly) \
                               .sqrt() 
        collectionmean = collectionmean.where(collectionmean.gte(1),1) \
                           .where(collectionmean.lte(0),0) 
        if len(m.layers)>1:
            m.remove_layer(m.layers[1])
        m.add_layer(TileLayer(url=GetTileLayerUrl( collectionmean.visualize(min=0, max=1.0,opacity = 1))))
    except Exception as e:
        w_text.value =  'Error: %s'%e

w_run.on_click(on_run_button_clicked)

def on_preview_button_clicked(b):
    global result,m,cmap,smap,fmap,bmap,w_changemap
    jet = 'black,blue,cyan,yellow,red'
    smap = ee.Image(result.get('smap')).byte()
    cmap = ee.Image(result.get('cmap')).byte()
    fmap = ee.Image(result.get('fmap')).byte() 
    bmap = ee.Image(result.get('bmap')).byte() 
    opacity = w_opacity.value
    if w_changemap.value=='First':
        mp = smap 
        mx = count
    elif w_changemap.value=='Last':
        mp=cmap
        mx = count
    else:
        mp = fmap
        mx = count/2
    if len(m.layers)>1:
        m.remove_layer(m.layers[1])
    m.add_layer(TileLayer(url=GetTileLayerUrl( mp.visualize(min=0, max=mx, palette=jet,opacity = opacity))))
    w_export.disabled = False
    
w_preview.on_click(on_preview_button_clicked)   

def on_export_button_clicked(b):
    global w_exportname
    point0 = ee.Geometry.Point(ee.List(poly.bounds().coordinates().get(0)).get(0))
    point1 = ee.Geometry.Point(ee.List(poly.bounds().coordinates().get(0)).get(1))
    point2 = ee.Geometry.Point(ee.List(poly.bounds().coordinates().get(0)).get(2))
    point3 = ee.Geometry.Point(ee.List(poly.bounds().coordinates().get(0)).get(3))
    collection1 = ee.ImageCollection('COPERNICUS/S2') \
                    .filterBounds(poly) \
                    .filterDate(ee.Date(w_startdate.value),ee.Date(w_enddate.value)) \
                    .sort('CLOUDY_PIXEL_PERCENTAGE',True) \
                    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',1.0) 
    count1 = collection1.size().getInfo()
    if count1>0:
#      use sentinel-2 as background                        
        background = ee.Image(collection1.first()) \
                               .clip(poly) \
                               .select('B8') \
                               .divide(5000)    
    else:
#      use temporal averaged sentinel-1                                                                        
        background = collectionmean          
    cmaps = ee.Image.cat(cmap,smap,fmap,bmap,background).rename(['cmap','smap','fmap']+timestamplist1[1:]+['background'])                
    assexport = ee.batch.Export.image.toAsset(cmaps,
                                description='assetExportTask', 
                                assetId=w_exportname.value,scale=10,maxPixels=1e9)
    assexportid = str(assexport.id)
    w_text.value= 'Exporting to %s\n task id: %s'%(w_exportname.value,assexportid)
    assexport.start()  
    
w_export.on_click(on_export_button_clicked)  

def run():
    global m,dc,center
    center = list(reversed(poly.centroid().coordinates().getInfo()))
    m = Map(center=center, zoom=10, layout={'height':'400px'})
    osm = basemap_to_tiles(basemaps.OpenStreetMap.HOT)
    mb = TileLayer(url="https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v9/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1IjoibWNhbnR5IiwiYSI6ImNpcjRsMmJxazAwM3hoeW05aDA1cmNkNzMifQ.d2UbIugbQFk2lnU8uHwCsQ")
    sm_control = SplitMapControl(left_layer=osm,right_layer=mb)
    m.add_control(dc)
    m.add_control(sm_control)
    display(m)
    display(box)
    