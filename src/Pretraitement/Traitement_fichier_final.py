import  pandas as pd
# pandas version 1.1.3
#import geopandas
import numpy as np
#from shapely.geometry import  Polygon,Point
import dask.array as da
import h5py
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
import time

pbar = ProgressBar()
pbar.register()

'''Pour le donne d'entrainement'''
def training_fichier(input_data,ind):
    ddf = dd.read_csv(input_data,header=None,sep = ' ')
    ddf = ddf.rename(columns={0:'X',1:'Y',2:'Z',3:'I'})
    ddf['L'] = 0
    data_labelize = dd.merge(ddf,daska_data,on=['X','Y','Z'])
    data_labelize = data_labelize.drop(columns=['I_y','L_x'])
    data_labelize = data_labelize.rename(columns={'I_x':'I','L_y':'L'})
    data_labelize = dd.concat([ddf,data_labelize],ignore_order= True).drop_duplicates(subset = ['X','Y','Z'],keep = 'last' )
    data_labelize = data_labelize.reset_index(drop = True)
    data_train = data_labelize[data_labelize.X < 1422376.7884]
    dd.to_parquet(data_train,out_path.to_numpy()[ind][0])

'''Pour le donne du test/visualisation avec rgb'''
def test_fichier_labeliz(input_data,k):
    ddf = dd.read_csv(input_data,header = None,sep = ' ')
    ddf = ddf.rename(columns={0:'X',1:'Y',2:'Z',3:'I'})
    ddf['R'] = 0
    ddf['G'] = 255
    ddf['B'] = 0
    ddf['I'] = 0
    ddf['L'] = 0
    data_labelize = dd.merge(ddf,daska_data,on=['X','Y','Z'])
    data_labelize = data_labelize.drop(columns=['I_x','R_x','G_x','B_x'])
    data_labelize = data_labelize.rename(columns={'I_y':'I','R_y':'R','G_y':'G','B_y':'B'})
    data_labelize = dd.concat([ddf,data_labelize],ignore_order= True).drop_duplicates(subset = ['X','Y','Z'],keep = 'last' )
    data_labelize = data_labelize.reset_index(drop = True)
    data_test = data_labelize[data_labelize.X >= 1422376.7884]
    dd.to_parquet(data_test,out_path.to_numpy()[k][0])


'''charger tout les nuages de points labelize et filtre % à z'''
in_path = pd.read_csv("D:\VR3D\VR3D_DATASET\Output_nuage\Output_las\File_list.txt",header=None).to_numpy()

'''charger tout les 220 chambre'''
daska_data = pd.read_csv("D:\VR3D\VR3D_DATASET\Output_nuage\Lasdisk_file.txt",header=None,sep = ' ')#220 chambre
daska_data = daska_data.rename(columns= {0:'X',1:'Y',2:'Z',3:'I'})

'''pour le donne de test uniquement 
daska_data['R'] = 255
daska_data['G'] = 0
daska_data['B'] = 0
daska_data['I'] = 0'''

out_path = pd.read_csv("D:\VR3D\VR3D_DATASET\Out_train\OUT.txt",header = None)

''' Appeler fonction pour le train
for i in range(len(in_path)):
    training_fichier(in_path[i][0],i)'''

'''Appeler fonction pour le test
for i in range(len(in_path)):
    test_fichier_labeliz(in_path[i][0],i)'''