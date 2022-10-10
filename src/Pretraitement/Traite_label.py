import sys
import ezdxf
from  shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import ezdxf.math as math
from collections import Counter
import matplotlib.pyplot as plt
import  pandas as pd
# pandas version 1.1.3
lats_vect= []
lons_vect = []
import numpy as np
"""
dxf_files = pd.read_csv('D:\VR3D\VR3D_DATASET\J35_txt_cutted\\fichier_2D\dxf_file.txt',header = None)
for i in range(len(dxf_files)):
	print(dxf_files.loc[i][0])
	doc = ezdxf.readfile(str(dxf_files.loc[i][0]))
	msp = doc.modelspace()
	lwschema = msp.query('LWPOLYLINE')
	for pline in lwschema:
		if pline.has_arc:
			print('\n\nHAS ARC = ', pline.has_arc)
		else:
			print('care/rectangle')
			lons, lats = [], []
			for p in range(len(pline)):
				x, y, s, e, b = pline[p]
				lons.append(y)
				lats.append(x)
		lats_vect.append(lats)
		lons_vect.append(lons)
print(len(lats_vect))

"""
doc = ezdxf.readfile("D:\VR3D\VR3D_DATASET\J35_txt_cutted\\fichier_2D\BORDEAUX_Travail_J35_ASS.dxf")
msp = doc.modelspace()
lwschema = msp.query('LWPOLYLINE')
for pline in lwschema :
	if not pline.has_arc:
		print('carre/rectangle')
		lons,lats = [],[]
		for p in range(len(pline)):
			x,y,s,e,b = pline[p]
			lons.append(y)
			lats.append(x)
	lats_vect.append(lats)
	lons_vect.append(lons)
print(lats_vect[216],lons_vect[216])
#del(lats_vect[219])
#del(lons_vect[219])
def balayage_care(lons_vect,lats_vect,nuage):#nuage:(y,x)
	lons_lats_vect = np.column_stack((lons_vect, lats_vect))  # Reshape coordinates
	polygon = Polygon(lons_lats_vect)  # create polygon
	point = Point(nuage)  # create point
	#print(polygon.contains(point))  # check if polygon contains point
	if polygon.contains(point):# check if a point is in the polygon
		return True
	else :
		return False

print('\n -------------------------- \n')

"""------------------------pour test le model----------------------
doc = ezdxf.readfile("data/DX_J35_ASS.dxf")
ex : nuage_points = pd.read_csv('D:\VR3D\VR3D_DATASET\Donner_split\disk60.txt',error_bad_lines = False,sep = ' ',header = None)+
output 219 former par le disk 1 --> 10.txt : 100 Mo
--> output == nuage_points_216.txt : --> 486 ko

"""
# lire nuage de point
#nuage_points = pd.read_csv("D:\VR3D\VR3D_DATASET\Donner_split\disk2.txt",error_bad_lines = False,sep = ' ',header = None)
def create_fichier_list():
	fichier_list = []
	for i in range(len(lats_vect)):
		fichier_list.append('D:\VR3D\VR3D_DATASET\output_nuage\\nuage_points_'+str(i)+'.txt')
	return  fichier_list
'''-------------------------pour tout les donne---------------------------'''
"""
for j in range (11):
	nuage_points = pd.read_csv("D:\VR3D\VR3D_DATASET\Donner_split\Disk_finau\Disk"+str(j+1)+".txt",error_bad_lines=False,sep=' ',header=None)
	print(len(nuage_points))
	for nu in range(len(nuage_points)):
		fichier_list = create_fichier_list()
		test = True
		i = 219
		#while test and  i < len(lats_vect) :
		while test and i < len(lats_vect):
			lats_vect_loc = np.append(lats_vect[i],lats_vect[i][0])
			lons_vect_loc = np.append(lons_vect[i],lons_vect[i][0])
			#print(balayage_care(lons_vect_loc,lats_vect_loc,(nuage_points.loc[nu][1],nuage_points.loc[nu][0])))

			if balayage_care(lons_vect_loc,lats_vect_loc,(nuage_points.loc[nu][1],nuage_points.loc[nu][0])):
				fichier = open(fichier_list[i],'a')
				fichier.write(str(nuage_points.loc[nu][0])+' '+str(nuage_points.loc[nu][1])+' '
						  +str(nuage_points.loc[nu][2])+' '+str(nuage_points.loc[nu][3])+'\n')
				fichier.close()
				test = False
				print(nu)
				#print('-----------------------------en voie un nuage de points-------------------------------------')
			else :
				i+=1
"""
i = 0
for nu in range(len(nuage_points)):
	fichier_list = create_fichier_list()
	test = True
	while test and i < len(lats_vect):
		lats_vect_loc = np.append(lats_vect[i],lats_vect[i][0])
		lons_vect_loc = np.append(lons_vect[i],lons_vect[i][0])
		#print(balayage_care(lons_vect_loc,lats_vect_loc,(nuage_points.loc[nu][1],nuage_points.loc[nu][0])))

		if balayage_care(lons_vect_loc,lats_vect_loc,(nuage_points.loc[nu][1],nuage_points.loc[nu][0])):
			fichier = open(fichier_list[i],'a')
			fichier.write(str(nuage_points.loc[nu][0])+' '+str(nuage_points.loc[nu][1])+' '
					  +str(nuage_points.loc[nu][2])+' '+str(nuage_points.loc[nu][3])+'\n')
			fichier.close()
			test = False
			print(nu)
			#print('-----------------------------en voie un nuage de points-------------------------------------')
		else :
			i+=1