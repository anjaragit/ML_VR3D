import ezdxf
from  shapely.geometry import Point
from shapely.geometry.polygon import Polygon,Point
import ezdxf.math as math
from collections import Counter
import matplotlib.pyplot as plt
import  pandas as pd
# pandas version 1.1.3
import geopandas
import numpy as np
import time
import dask.dataframe as dd
from dask.diagnostics import ProgressBar



pbar = ProgressBar()
pbar.register()


def dxf_to_geopandas():
    doc = ezdxf.readfile("D:\VR3D\VR3D_DATASET\J35_txt_cutted\\fichier_2D\BORDEAUX_Travail_J35_ASS.dxf")
    msp = doc.modelspace()
    lwschema = msp.query('LWPOLYLINE')
    polys1 = geopandas.GeoSeries()
    polys2 = geopandas.GeoSeries()
    pol = []
    Rayon = []
    print(len(lwschema))
    for pline in lwschema :
        if not pline.has_arc:
            poly = []
            ma_x = []
            ma_y = []
            for p in range(len(pline)):
                x,y,s,e,b = pline[p]
                ma_x.append(x)
                ma_y.append(y)
                poly.append((x,y))
            pol.append([min(ma_x),min(ma_y),max(ma_x),max(ma_y)])
            #print(poly)
            #pol.append(poly)
            polys1 = polys1.append(geopandas.GeoSeries([Polygon(poly)]),ignore_index = True)
        else :
            polygon_data = {}
            bulg = []
            cx = []
            cy = []
            radius = []
            nxt = 0

            for pl in range(len(pline)):
                x, y, s, e, b = pline[pl]
                bulg.append(b)
                polygon_data[pl] = [x, y]

                if nxt > 0 :
                    C = math.bulge_center(polygon_data[nxt-1],polygon_data[nxt],bulg[nxt-1])
                    cx.append(C[0])
                    cy.append(C[1])
                    R = math.bulge_radius(polygon_data[nxt - 1], polygon_data[nxt], bulg[nxt - 1])
                    radius.append(R)

                if pl+1 == len(pline) :
                    centerX = np.mean(cx)
                    centerY = np.mean(cy)
                    cercleR = np.mean(radius)
                    Rayon.append(cercleR)
                    #AllCercle.append([centerX,centerY,cercleR])
                    cx = []
                    cy = []
                    radius = []
                nxt += 1
            pol.append([centerX - cercleR,centerY - cercleR,centerX + cercleR,centerY + cercleR])
            polys1 = polys1.append(geopandas.GeoSeries([Point(centerX,centerY).buffer(cercleR)]),ignore_index = True)
    return pol,polys1

def cherche_dask(chambre):
    df1 = ddf[(ddf.X > pg[chambre][0])&(ddf.X < pg[chambre][2])&(ddf.Y > pg[chambre][1])&(ddf.Y < pg[chambre][3])]
    return df1

def create_fichier_list():
    fichier_list = []
    for i in range(len(poly_geopandas)):
        fichier_list.append('D:\VR3D\VR3D_DATASET\OUT_220\\Nuage_points_'+str(i)+'.txt')
    return  fichier_list

def train(couple):
    for i in range(len(couple)):
        df = pd.DataFrame(poly_geopandas.contains(Point(couple.loc[i][0],couple.loc[i][1])))
        index = df.index
        condition = df[0] == True
        ind = index[condition]
        if not ind.empty:
            couple['L'][i] = 1
            fichier_list = create_fichier_list()
            fichier = open(fichier_list[ind[0]],'a')
            fichier.write(str(couple.loc[i][0])+' '+str(couple.loc[i][1])+' '
                      +str(couple.loc[i][2])+' '+str(couple.loc[i][3])+' '+str(couple.loc[i][4])+'\n')
            fichier.close()
    return couple

def final_train():
    for i in range(218):
        chre = cherche_dask(i)
        data_final = chre.compute()
        data_final = data_final.reset_index(drop = True)
        train(data_final)
        print('--------------------',i,'-----------------')



pg,poly_geopandas = dxf_to_geopandas()
poly_geopandas = poly_geopandas[:219]

'''Pour le gros data'''#ddf : dask dataframe
ddf = dd.read_csv("D:\VR3D\VR3D_DATASET\First_vr3d_PTS.txt",header=None,sep = ' ')
ddf = ddf.rename(columns={0:'X',1:'Y',2:'Z',3:'I'})
ddf['L'] = 0
'''appel final'''
final_train()