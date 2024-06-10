import os
import numpy as np
import matplotlib.pyplot as plt


def plotMap(llh, zoom_factor = 120, mapNo=1):
    DirM = os.getcwd() + '/'
    mapbbox = np.loadtxt(DirM + 'scripts/plotlib/Map/Mapfig/mapinfo.txt')
    map1 = plt.imread(DirM + 'scripts/plotlib/Map/Mapfig/Map{}.png'.format(mapNo))

    lat = llh[0, :]
    lon = llh[1, :]

    lon_l = mapbbox[mapNo-1,1]-mapbbox[mapNo-1,0]
    lat_l = mapbbox[mapNo-1,3]-mapbbox[mapNo-1,2]
    pix_per_lon_l = map1.shape[1]/lon_l
    pix_per_lat_l = map1.shape[0]/lat_l


    c = 0.001
    llzoom = np.array([np.min(lon)-c*zoom_factor,np.max(lon)+c*zoom_factor,np.min(lat)-c*zoom_factor,np.max(lat)+c*zoom_factor])


    zbbox = np.array([[llzoom[0],llzoom[2]],[llzoom[0],llzoom[3]],[llzoom[1],llzoom[3]],[llzoom[1],llzoom[2]],[llzoom[0],llzoom[2]]])
    llcrop_pix = np.array([(llzoom[0]-mapbbox[mapNo-1,0])*pix_per_lon_l,(llzoom[1]-mapbbox[mapNo-1,0])*pix_per_lon_l,
                        (-llzoom[3]+mapbbox[mapNo-1,3])*pix_per_lat_l,(-llzoom[2]+mapbbox[mapNo-1,3])*pix_per_lat_l]).astype(int)

    map_cropped = map1[llcrop_pix[2]:llcrop_pix[3],llcrop_pix[0]:llcrop_pix[1]]

    return llcrop_pix, llzoom, zbbox, map_cropped

