import sys
import ezdxf
import numpy
import pandas as pd
import ezdxf.math as math
from collections import Counter
from matplotlib import pyplot as plt

doc = ezdxf.readfile("data/DX_J35_ASS.dxf")

print(doc)
print('\n -------------------------- \n')

def centroid(vertexes):
	x_list = [vertex [0] for vertex in vertexes]
	y_list = [vertex [1] for vertex in vertexes]
	long = len(vertexes)
	x = sum(x_list) / long
	y = sum(y_list) / long
	return(x, y)

coord = []
mylist = []
myClass = []

msp = doc.modelspace()
lwschema = msp.query('LWPOLYLINE')
print(lwschema.count('DT-CH-SABOM'))

def isInsideCircle(cx, cy, r, x, y):
	dist = (x - cx) * (x - cx) + (y - cy) * (y - cy)
	if (dist <= r * r):
		print('Yes.')
		return True
	else:
		print('-')
		return False

AllCercle =[]

for pline in lwschema:

	if pline.has_arc :
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
				centerX = numpy.mean(cx)
				centerY = numpy.mean(cy)
				cercleR = numpy.mean(radius)
				AllCercle.append([centerX,centerY,cercleR])
				cx = []
				cy = []
				radius = []
			nxt += 1
	#else:
		#print('-Not-')
		#for pl in range(len(pline)):
		#	x, y, s, e, b = pline[pl]

print('\n -------------------------- \n')
#check if a point belong to circle
print('All Cercle = ',len(AllCercle))

# lire nuage de point
nuage_points = pd.read_csv('rcp_2/disk1.txt',error_bad_lines = False,sep = ' ',header = None)
print('load file...')

for i in range(int(len(AllCercle))):
	cX = AllCercle[i][0]
	cY = AllCercle[i][1]
	R = AllCercle[i][2]
	print('\n\nPour R =', R)

	for nu in range(int(len(nuage_points))):

		x = nuage_points.loc[nu][0]
		y = nuage_points.loc[nu][1]
		print('\tx=', x, ' y=', y)

		if isInsideCircle(cX, cY, R, x, y):
			fout = open('data_out/nuage_out.txt', 'w')
			fout.write(str(nuage_points.loc[nu][0])+' '+str(nuage_points.loc[nu][1])+' '
							  +str(nuage_points.loc[nu][2])+' '+str(nuage_points.loc[nu][3])+'\n')
			fout.close()
			print( x,y, ' Inside..' )
