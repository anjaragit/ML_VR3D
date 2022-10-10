import numpy as np
import time
from laspy.file import File
from dask import delayed
import dask
import warnings
from scipy import stats
warnings.filterwarnings("ignore")

# -*- coding: Utf-8 -*-
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

    bin_x = np.linspace(int(x_min), int(x_max),int(nbr_bin_x))
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


def calcl_dist(zone_1):
    #------------------calculer distance entre deux points-------------------------- #
    kmin_1 = np.min(zone_1[:, 0])
    kmax_1 = np.max(zone_1[:, 0])
    dist = kmax_1 - kmin_1
    return dist


def process_intensity(array):
    """
    translation de l'intensite vers le plage (13120.0 ; 48377.0 )
    :param array:
    :return:
    """
    if len(array)!=0:
        # data = pd.read_csv('D:\\VR3D\code\\ML-VR3D\\code\\ML-VR3D\\Pretraitement\\data\\out2.txt')
        # array = np.array(data)
        array = array[(array[:, 3] >= np.quantile(array[:, 3], 0.1)) & (array[:, 3] <= np.quantile(array[:, 3], 0.99))]
        min_arr = min(array[:, 3])
        max_arr = max(array[:, 3])

        # 13120.0 = a * min_arr + b
        # 48377.0 = a * max_arr + b
        a = (13120.0 - 48377.0)/(min_arr - max_arr)
        b = 13120.0 - a * min_arr
        array[:, 3] = array[:, 3] * a + b
        print('stat describe', stats.describe(array[:, 3]))
    return array

def decouper_zone(fil_name):
    REF_DECOUP = 150
    data_in = File(fil_name,mode='r')
    binary_classes = np.empty((len(data_in.x), 4))
    binary_classes[:, 0] = data_in.x
    binary_classes[:, 1] = data_in.y
    binary_classes[:, 2] = data_in.z
    binary_classes[:, 3] = data_in.intensity

    # -------------------------------equilibre intensity----------------------------- #
    binary_classes = process_intensity(binary_classes)
    # print(stats.describe(binary_classes[:, 3]))
    # -------------------------------decouper zone----------------------------------- #
    zone_1 = binary_classes[binary_classes[:, 0] <= np.mean(binary_classes[:, 0])]
    # zone_2 = binary_classes[binary_classes[:, 0] > np.mean(binary_classes[:, 0])]
    zone = []
    dist = calcl_dist(zone_1)
    '''incr = 0
    while dist > REF_DECOUP:
        zone_1 = zone_1[zone_1[:, 0] <= np.mean(zone_1[:, 0])]
        dist = calcl_dist(zone_1)
        incr +=1'''
    # tester si le difference de kilometrage % l'axe x soit plus petit que 120km
    if dist > REF_DECOUP:
        zone.append(zone_1[zone_1[:, 0] <= np.mean(zone_1[:, 0])])
        zone.append(zone_1[zone_1[:, 0] > np.mean(zone_1[:, 0])])
        zone_2 = binary_classes[binary_classes[:, 0] > np.mean(binary_classes[:, 0])]
        zone.append(zone_2[zone_2[:, 0] <= np.mean(zone_2[:, 0])])
        zone.append(zone_2[zone_2[:, 0] > np.mean(zone_2[:, 0])])
    else :
        zone.append(zone_1)
        zone.append(binary_classes[binary_classes[:, 0] > np.mean(binary_classes[:, 0])])
    # outputFile = open(filtered_name, "w")
    # for index in range(len(new_binary_classes)):
    #    outputFile.write(str(new_binary_classes[index, 0]) + " " + str(new_binary_classes[index, 1])
    #                     + " " + str(new_binary_classes[index, 2])
    #                                   + " " + str(new_binary_classes[index, 3]) + "\n")
    # outputFile.close()
    return zone


def run(zone, i):
    pos_bin_x, pos_bin_y = binning_array(7.5, zone[i])
    return filter_z_local(zone[i], pos_bin_x, pos_bin_y)

def run_filter(fil_name):
    """

    :param fil_name: original_filename
    :return: data_numpy filtre
    """
    zone = decouper_zone(fil_name)
    zone_array = []
    for i in range(len(zone)):
        zone_array.append(delayed(run)(zone, i))
        # pos_bin_x, pos_bin_y = binning_array(7.5, zone[i])
        # zone_array.append(filter_z_local(zone[i], pos_bin_x, pos_bin_y))
    zone_array = dask.compute(*zone_array)
    new_binary_classes = np.concatenate([i for i in zone_array],axis=0)
    '''
    outputFile=open(filtered_name+'_filtre.txt',"w")
    t2=time.time()-t
    print("Temps de traitement sans las", t2, "secondes")
    for index in range(len(new_binary_classes)):
        outputFile.write(str(new_binary_classes[index,0]) + " " + str(new_binary_classes[index,1])
                             + " " + str(new_binary_classes[index,2]) + " " + str(new_binary_classes[index,3]) + "\n")
    outputFile.close()
    t1=time.time()-t
    print("traitement en",t1,"secondes" )
    print("Generating las FINISH ...")
    # os.system('wine lasview.exe -i '+ filtered_name +' -iparse xyzi')'''
    return new_binary_classes

if __name__ == "__main__":
    t1 = time.time()
    run_filter('D:\ML_Polyligne_Detection\data\Z3.las')
    t_final = time.time()
    print('time execution in ',t_final - t1)