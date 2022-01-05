'''
Created on 08.04.2019

@author: mort

ipywidget interface to the GEE for UNet

'''
import ee
import ipywidgets as widgets
from IPython.display import display
from ipyleaflet import (Map,DrawControl,TileLayer,
                        basemaps,basemap_to_tiles,
                        LayersControl,
                        MeasureControl,
                        FullScreenControl)
from geopy.geocoders import Nominatim

ee.Initialize()

geolocator = Nominatim(timeout=10,user_agent='interface.ipynb')

def makefeature(data):
    ''' for exporting as CSV to Drive '''
    return ee.Feature(None, {'data': data})

def handle_draw(self, action, geo_json):
    global poly
    if action == 'created': 
        coords =  geo_json['geometry']['coordinates']
        poly = ee.Geometry.Polygon(coords)
        w_collect.disabled = False
        w_export.disabled = False
        
dc = DrawControl(polyline={},circlemarker={})
dc.rectangle = {"shapeOptions": {"fillColor": "#0000ff","color": "#0000ff","fillOpacity": 0.1}}
dc.polygon = {"shapeOptions": {"fillColor": "#0000ff","color": "#0000ff","fillOpacity": 0.1}}

dc.on_draw(handle_draw)

def GetTileLayerUrl(ee_image_object):
    map_id = ee.Image(ee_image_object).getMapId()
    return map_id["tile_fetcher"].url_format

w_collection = widgets.RadioButtons(
    options=['USDA/NAIP/DOQQ','SKYSAT/GEN-A/PUBLIC/ORTHO/MULTISPECTRAL'],
    value='USDA/NAIP/DOQQ',
    layout=widgets.Layout(width='90%', height='80px'),
    description='Collection:',
    disabled=False
)

w_startdate = widgets.Text(
    value='2014-01-01',
    placeholder=' ',
    description='Start date:',
    disabled=False
)
w_enddate = widgets.Text(
    value='2018-12-31',
    placeholder=' ',
    description='End date:',
    disabled=False
)
w_text = widgets.Textarea(
    layout = widgets.Layout(width='75%'),
    value = 'Output',
    rows = 3,
    disabled = False
)
w_location = widgets.Text(
    value='Philadelphia',
    placeholder=' ',
    description='',
    disabled=False
)
w_scale = widgets.FloatText(
    layout = widgets.Layout(width='150px'),
    value=1,
    placeholder=' ',
    description='Scale ',
    disabled=False
)
w_exportname = widgets.Text(
    value='<filename>',
    placeholder=' ',
    disabled=False
)

w_goto = widgets.Button(description='GoTo')
w_collect = widgets.Button(description="Collect",disabled=True)
w_export = widgets.Button(description='Export to Drive',disabled=True)
w_scalesig = widgets.HBox([w_scale])
w_exp = widgets.HBox([w_export,w_exportname])
w_top = widgets.HBox([w_text,w_goto,w_location])
w_mid = widgets.HBox([w_startdate,w_enddate])
w_bot = widgets.HBox([w_collect,w_exp,w_scale])

box = widgets.VBox([w_top,w_collection,w_mid,w_bot])

def on_widget_change(b):
    pass
    
w_collection.observe(on_widget_change,names='value')

def on_goto_button_clicked(b):
    try:
        location = geolocator.geocode(w_location.value)
        m.center = (location.latitude,location.longitude)
        m.zoom = 10
    except Exception as e:
        w_text.value =  'Error: %s'%e

w_goto.on_click(on_goto_button_clicked)

def rgbLayer(image):
    ''' two percent linear stretch '''
    rgbim = image.rename('r','g','b')
    ps = rgbim.reduceRegion(ee.Reducer.percentile([2,98]),scale=10,maxPixels=1e10).getInfo()
    mx = [ps['r_p98'],ps['g_p98'],ps['b_p98']]
    mn = [ps['r_p2'],ps['g_p2'],ps['b_p2']]
    return rgbim.visualize(min=mn,max=mx)

def on_collect_button_clicked(b):
    global hr
    try:
        w_text.value = 'Collecting ...'   
        hr = ee.ImageCollection(w_collection.value) \
                      .filterDate(ee.Date(w_startdate.value), ee.Date(w_enddate.value)) \
                      .filterBounds(poly) \
                      .select(['R','G','B','N']) \
                      .mosaic() \
                      .clip(poly)                                                                                        
        layer = TileLayer(url=GetTileLayerUrl(rgbLayer(hr.select(['R','G','B']))))
        if len(m.layers)>2:
            m.remove_layer(m.layers[2])              
        m.add_layer(layer)            
        w_text.value = 'Done'                
    except Exception as e:
        w_text.value =  'Error: %s'%e

w_collect.on_click(on_collect_button_clicked)

def on_export_drv_button_clicked(b):
    try:
        fileNamePrefix =  w_exportname.value
        gdexport = ee.batch.Export.image.toDrive(hr,
                                    description='driveExportTask', 
                                    folder = 'gee',
                                    crs = 'EPSG:26916',
                                    fileNamePrefix=fileNamePrefix,scale=w_scale.value,maxPixels=1e11)   
        gdexport.start()
        w_text.value = 'Exporting aoi to Drive/gee/%s\n task id: %s'%(fileNamePrefix,str(gdexport.id)) 
    except Exception as e:
        w_text.value =  'Error: %s'%e    
    
w_export.on_click(on_export_drv_button_clicked)   

def run():
    global m, lc, osm
    center = [37.26,-93.31]
    osm = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
    ewi = basemap_to_tiles(basemaps.Esri.WorldImagery)
    lc = LayersControl(position='topright')
    fs = FullScreenControl(position='topleft')
    mc = MeasureControl(position='topright',primary_length_unit='kilometers')
    m = Map(center=center, zoom=10, layout={'height':'400px'},layers=(osm,ewi),controls=(mc,dc,lc,fs))   
    display(m) 
    return box
    