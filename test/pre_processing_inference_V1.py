# -*- coding: Utf-8 -*-
import os
import glob
from Filter_local import *
import warnings
warnings.filterwarnings("ignore")

def pre_process(data_path, collect_path):
    """
    
    :param data_path:
    :param collect_path:
    :return:
    """
    # if not os.path.exists(collect_path):
    #    os.mkdir(collect_path)

    for f in glob.glob(os.path.join(data_path, '*.las')):
        # Récupérer nom du fichier
        elements = f.split('/')
        elements2 = elements[1].split('\\')
        filename = elements2[-1]
        elements2 = filename.split('.')
        out_filename = elements2[0]
        print('###############', out_filename)
        file_write = os.path.join(collect_path, out_filename)
        print('###############', file_write)

        filtered_data = run_filter(f)

        # filtre du nuage de point

        #xyz_min = np.amin(filtered_data, axis=0)[0:3]
        #filtered_data[:, 0:3] -= xyz_min
        #fmin = open(os.path.join(collect_path, 'xyz_min.txt'), 'a')
        #fmin.write('%f %f %f \n' % (xyz_min[0], xyz_min[1], xyz_min[2]))
        #fmin.close()

        # Sauvegarde les fichiers numpy dans collect_path
        np.save(file_write, filtered_data)

        # Append numpy_file_list
        f_list = open(os.path.join(collect_path, 'numpy_file_list.txt'), 'w')
        f_list.write(collect_path+'/'+out_filename + '.npy' + '\n')
        f_list.close()
        return filtered_data
# pre_process('test/Annotations/','test/collect')