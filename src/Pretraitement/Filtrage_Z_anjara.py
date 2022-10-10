import os
import pandas as pd
import numpy as np
import time
from laspy.file import File

'''Filtrage brute sans utilise de transformation en  fichier las'''

def z_filter(lasfile, med_ref):
    listZ = lasfile[:, 2]
    # ratio = len(listZ) *0.35
    # med_ref=0
    min_z = listZ.min()
    max_z = listZ.max()
    e = max_z - min_z
    med = np.median(listZ)
    e_med = med_ref - min_z
    if med_ref < med:
        if e / 2 <= e_med:
            new_medr = med
        else:
            new_medr = med_ref
    else:
        new_medr = med
    std_z = max(np.std(listZ), 0.01)
    if std_z < 1:
        compare_min = new_medr - 0.75
        compare_max = new_medr + 1.9
    else:
        compare_min = new_medr - 0.75
        compare_max = new_medr + 1.76
    indice_filter_array = np.where((compare_max > listZ) & (listZ > compare_min))

    return lasfile[indice_filter_array], new_medr


def binning_array(delta, binary_classes):
    x_min = min(binary_classes[:, 0])
    x_max = max(binary_classes[:, 0])
    y_min = min(binary_classes[:, 1])
    y_max = max(binary_classes[:, 1])

    z_min = min(binary_classes[:, 2])
    z_max = max(binary_classes[:, 2])
    nbr_bin_x = np.ceil((x_max - x_min) / delta)
    nbr_bin_y = np.ceil((y_max - y_min) / delta)

    bin_x = np.linspace(x_min, x_max,int(nbr_bin_x))
    bin_y = np.linspace(y_min, y_max, int(nbr_bin_y))

    pos_bin_x = np.digitize(binary_classes[:, 0], bin_x)
    pos_bin_y = np.digitize(binary_classes[:, 1], bin_y)
    return pos_bin_x, pos_bin_y


def filter_z_local(binary_classes, pos_bin_x, pos_bin_y):
    n_c = 0
    bin_z = binary_classes[:, 2]
    new_binary_classes = []
    med_bloc = []
    nbr_bin_x = len(np.unique(pos_bin_x))
    nbr_bin_y = len(np.unique(pos_bin_y))
    medr = max(bin_z)
    print("nbr_bin_x", nbr_bin_x)
    print("nbr_bin_y", nbr_bin_y)
    for i in range(int(nbr_bin_y)):
        for j in range(int(nbr_bin_x)):
            indices_i_j = np.where((pos_bin_x == j + 1) & (pos_bin_y == i + 1))
            if (np.array(indices_i_j)).size == 0:
                n_c += 1
            else:
                array_bin = binary_classes[indices_i_j]
                array_bin_filter, medr = z_filter(array_bin, medr)
                new_binary_classes.append(array_bin_filter)
    print("Nombre de bin totale", nbr_bin_y * nbr_bin_x)
    print("Nombre de bin sans points", n_c)
    print(new_binary_classes)
    new_binary_classes = np.concatenate(new_binary_classes, axis=0)
    return new_binary_classes


def run_filter(fil_name,filtered_name):  # args: filtered_name
    t = time.time()
    print ('############################', fil_name)
    data_in = pd.read_csv(fil_name,header = None,sep = ' ')
    binary_classes = np.empty((len(data_in[0]), 4))
    binary_classes[:, 0] = data_in[0].to_numpy()
    binary_classes[:, 1] = data_in[1].to_numpy()
    binary_classes[:, 2] = data_in[2].to_numpy()
    binary_classes[:, 3] = data_in[3].to_numpy()

    # binary_classes=pd.read_csv(fil_name, delimiter=" ", header=None).values

    pos_bin_x, pos_bin_y = binning_array(7.5, binary_classes)
    new_binary_classes = filter_z_local(binary_classes, pos_bin_x, pos_bin_y)
    print("Avant filtre:", len(binary_classes))
    print("Apres filtre:", len(new_binary_classes))
    #return new_binary_classes
    outputFile=open(filtered_name,"w")
    t2=time.time()-t
    print("Temps de traitement sans las", t2, "secondes")
    for index in range(len(new_binary_classes)):
            outputFile.write(str(new_binary_classes[index,0]) + " " + str(new_binary_classes[index,1]) + " " + str(new_binary_classes[index,2]) + " " + str(new_binary_classes[index,3]) + "\n")
    outputFile.close()
    t1=time.time()-t
    print("traitement en",t1,"secondes" )
    print("Generating las FINISH ...")
    #os.system('wine lasview.exe -i '+ filtered_name +' -iparse xyzi')


input_file = pd.read_csv("D:\VR3D\VR3D_DATASET\Donner_split\Disk_split\File_list.txt",header=None)
output_file = pd.read_csv("D:\VR3D\VR3D_DATASET\Output_nuage\Output_las\File_list.txt",header=None)
for i in range(len(input_file)):
    run_filter(input_file[0][i],output_file[0][i])