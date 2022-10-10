'''Genere de fichier parquet to las '''
import pandas as pd
import os
'''
Genere une liste de fichier part..parquet s'il n'y a pas au debut
list_file = pd.DataFrame()
for i in range(75):#nombre de fichier parquet genere apres filter_z
    list_file = pd.concat([list_file,pd.DataFrame(['D:\VR3D\VR3D_DATASET\Out_test\part.'+str(i)+'.parquet'])])
list_file.to_csv('D:\VR3D\VR3D_DATASET\Out_test\List_file.txt',index=False,header=None)
'''

fichier_input = pd.read_csv('D:\\VR3D\\VR3D_DATASET\\Out_test_labelize\\List_file.txt',header=None)
#save_path = 'D:\\VR3D\\VR3D_DATASET\\Out_test_labelize\\Las_file_rgb.txt'
def save_to_las_rgb(j):
    pc_label = pd.read_parquet(fichier_input[0][j]).to_numpy()
    fout = open(save_path_rgb, 'a')
    for i in range(pc_label.shape[0]):
        fout.write('%f %f %f %d %d %d %d\n' % \
                   (pc_label[i, 0], pc_label[i, 1], pc_label[i, 2],
                    pc_label[i, 3], pc_label[i, 4], pc_label[i, 5],pc_label[i, 6]))
    fout.close()
    print('OK... ',j)
def save_to_las(j):
    pc_label = pd.read_parquet(fichier_input[0][j]).to_numpy()
    fout = open(save_path, 'a')
    for i in range(pc_label.shape[0]):
        fout.write('%f %f %f %d\n' % \
                   (pc_label[i, 0], pc_label[i, 1], pc_label[i, 2],
                    pc_label[i, 3]))
    fout.close()
    print('OK... ',j)


save_path = 'D:\\VR3D\\VR3D_DATASET\\Out_test\\Las_file_3.txt'
save_path_rgb = 'D:\\VR3D\\VR3D_DATASET\\Out_test_labelize\\Las_file_3.txt'

'''pour donner de test sans rgb
for i in range(len(fichier_input)):
    save_to_las(i)
os.system('D:\VR3D\Tools\LAStools\\bin\\txt2las.exe -i \"' + save_path + '\" -parse xyzi -olas')'''



'''pour donner de test a visualize avec rgb 
for i in range(len(fichier_input)):
    save_to_las_rgb(i)
os.system('D:\VR3D\Tools\LAStools\\bin\\txt2las.exe -i \"' + save_path_rgb + '\" -parse xyzi -olas')
'''