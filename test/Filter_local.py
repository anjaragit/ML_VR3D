# -*- coding: Utf-8 -*-
import numpy as np
import time
import pandas as pd
from laspy.file import File


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

    indice_filter_array = np.where(
        (compare_max > listZ) & (listZ > compare_min))

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

    bin_x = np.linspace(int(x_min), int(x_max), int(nbr_bin_x))
    bin_y = np.linspace(int(y_min), int(y_max), int(nbr_bin_y))

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
    for i in range(int(nbr_bin_y)):
        for j in range(int(nbr_bin_x)):
            indices_i_j = np.where((pos_bin_x == j + 1) & (pos_bin_y == i + 1))
            if (np.array(indices_i_j)).size == 0:
                n_c += 1
            else:
                array_bin = binary_classes[indices_i_j]
                array_bin_filter, medr = z_filter(array_bin, medr)
                new_binary_classes.append(array_bin_filter)
    new_binary_classes = np.concatenate(new_binary_classes, axis=0)
    return new_binary_classes


def run_filter(fil_name):  # args: filtered_name
    t = time.time()
    # print ('############################', str(fil_name))
    # lasfile = File(fil_name, mode='rw')
    '''binary_classes = np.empty((len(lasfile.x), 4))

    binary_classes[:, 0] = lasfile.x
    binary_classes[:, 1] = lasfile.y
    binary_classes[:, 2] = lasfile.z
    binary_classes[:, 3] = lasfile.intensity'''
    binary_classes = fil_name.copy()
    # binary_classes = '%s' % fil_name
    # equilibre intensite
    # binary_classes = process_intensity(binary_classes)
    #binary_classes = pd.read_csv(fil_name, delimiter=" ", header=None).values

    pos_bin_x, pos_bin_y = binning_array(7.5, binary_classes)
    new_binary_classes = filter_z_local(binary_classes, pos_bin_x, pos_bin_y)
    return new_binary_classes


def process_intensity(array):
    """
    translation de l'intenstite vers le plage (13120.0 ; 48377.0)
    :param array:
    :return:
    """
    array = array[(array[:, 3] >= np.quantile(array[:, 3], 0.1)) &
                  (array[:, 3] <= np.quantile(array[:, 3], 0.99))]
    min_arr = min(array[:, 3])
    max_arr = max(array[:, 3])

    # 13120.0 = a * min_arr + b
    # 48377.0 = a * max_arr + b
    a = (13120.0 - 48377.0)/(min_arr - max_arr)
    b = 13120.0 - a * min_arr
    array[:, 3] = array[:, 3] * a + b
    return array


'''
	outputFile=open(filtered_name,"w")
	t2=time.time()-t
	print("Temps de traitement sans las", t2, "secondes")
	for index in range(len(new_binary_classes)):
			outputFile.write(str(new_binary_classes[index,0]) + " " + str(new_binary_classes[index,1]) + " " + str(new_binary_classes[index,2]) + " " + str(new_binary_classes[index,3]) + "\n")
	outputFile.close()
	t1=time.time()-t
	print("traitement en",t1,"secondes" )
	print("Generating las FINISH ...")
	os.system('wine lasview.exe -i '+ filtered_name +' -iparse xyzi')

path = "D:\VR3D\VR3D_DATASET\Out_nuage\Filtre_intensite\chambre_filtre154.las"
print(run_filter(path))
'''
