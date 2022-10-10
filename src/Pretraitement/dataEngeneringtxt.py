import pandas as pd
import numpy as np
# import os
# import time
import dask
import time
import argparse
from dask import delayed
import warnings
import geopandas
import h5py
import ezdxf
from scipy import stats
import ezdxf.math as math
from sklearn.cluster import KMeans
from filter_local import *
from shapely.geometry.polygon import Polygon,Point
from laspy.file import File
warnings.filterwarnings('ignore')


NUM_POINT = 4096
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 7]
# data_dim = [NUM_POINT, 7]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype=np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype=np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0  # state: the next h5 file to save

def parse_args():
    parser = argparse.ArgumentParser(description="Labelisation Automatique chambre")
    parser.add_argument(
        "--dxf_path",
        type=str,
        # default="D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_XX\DXF_file\Z1\ASS\PUY-DE-DOME_ASS_1-11.dxf",
        default="D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_Bordeau\DXF_file\Z1\ASS\BORDEAUX_Travail_J35_ASS.dxf",
        # default="D:\VR3D\code\ML-VR3D\code\ML-VR3D\Pretraitement\data\zone_file\Z6\PARIS_TELECOM_Z6_.dxf",
        help="Path dxf by topo",
    )
    parser.add_argument(
        "--nuage_path",
        type=str,
        # default="D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_XX\\1-11.las",
        default="D:\VR3D\code\ML-VR3D\code\ML-VR3D\Pretraitement\data\zone_file\Z7\Z7.las",
        help="Path to nuage de points topo",
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default="D:\VR3D\code\ML-VR3D\code\ML-VR3D\Pretraitement\data\h5_data\\train_VR3D_Zrect_pts_filtre_11_04_2022_H5",
        help="Path to data saved to h5 file format",
    )
    parser.add_argument(
        "--marge",
        type=float,
        default=0.5,
        help="marge de contour pour chaque sommet",
    )
    return parser.parse_args()

# ---------------------------scrape nuage pour chaque chambre ------------------------ #
def dxf_to_geopandas(dxf_file):
    """
    scraper tout les sommet du polyligne et transforme en format polygone
    :param dxf_file:file dxf by topo
    :return: point cle de chaque polyligne, sommet polyligne sous forme polygone, z_elevation pour chaque sommet
    """
    # doc = ezdxf.readfile(dxf_file[0][ind_dxf])
    doc = ezdxf.readfile(dxf_file)
    # doc = ezdxf.readfile("D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_paris\Download\PARIS_ASS_Z8.dxf")

    # print(dxf_file[0][ind_dxf])
    msp = doc.modelspace()
    lwschema = msp.query('LWPOLYLINE')

    polys1 = geopandas.GeoSeries()
    pol = []
    Rayon = []
    Z_eleva = []
    for pline in lwschema:
        Z_eleva.append(pline.dxf.elevation)
        if not pline.has_arc:
            poly = []
            ma_x = []
            ma_y = []
            for p in range(len(pline)):
                x, y, s, e, b = pline[p]
                ma_x.append(x)
                ma_y.append(y)
                poly.append((x, y))
            pol.append([min(ma_x), min(ma_y), max(ma_x), max(ma_y)])
            # pol.append(poly)
            # poly[0] = (poly[0][0]+10.00000,poly[0][1])
            polys1 = polys1.append(geopandas.GeoSeries([Polygon(poly)]), ignore_index=True)
        else:
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

                if nxt > 0:
                    C = math.bulge_center(polygon_data[nxt - 1], polygon_data[nxt], bulg[nxt - 1])
                    cx.append(C[0])
                    cy.append(C[1])
                    R = math.bulge_radius(polygon_data[nxt - 1], polygon_data[nxt], bulg[nxt - 1])
                    radius.append(R)

                if pl + 1 == len(pline):
                    centerX = np.mean(cx)
                    centerY = np.mean(cy)
                    cercleR = np.mean(radius)
                    Rayon.append(cercleR)
                    # AllCercle.append([centerX,centerY,cercleR])
                    cx = []
                    cy = []
                    radius = []
                nxt += 1
            pol.append([centerX - cercleR, centerY - cercleR, centerX + cercleR, centerY + cercleR])
            polys1 = polys1.append(geopandas.GeoSeries([Point(centerX, centerY).buffer(cercleR)]), ignore_index=True)
    return pol, polys1, Z_eleva


def cherche_dask(ind, df, pg):
    """

    :param ind: indice polyligne
    :param ddf: nuage de points
    :param pg: sommet cle polyligne
    :return: nuage de points de chaque polyligne sous forme carre
    """
    # ddf = ddf.rename(columns={0:'X',1:'Y',2:'Z',3:'I'})
    ddf = np.zeros((len(df), 5))
    ddf[:,:-1] = df
    df1 = ddf[(ddf[:, 0] >= pg[ind][0])&(ddf[:, 0] <= pg[ind][2])&(ddf[:, 1] >= pg[ind][1])&(ddf[:, 1] <= pg[ind][3])]

    return df1


def definitiv_chambre(couple, poly_geopandas):
    """

    :param couple:chambre sous forme carre
    :param poly_geopandas:polygone d'entourage de chaque chambre
    :return:chambre bien determiner
    """
    for i in range(len(couple)):
        df = pd.DataFrame(poly_geopandas.contains(Point(couple[i, 0], couple[i, 1])))
        index = df.index
        condition = df[0] == True
        ind = index[condition]
        if not ind.empty:
            couple[i, 4] = 1
            # fichier_list = create_fichier_list(poly_geopandas, ind_zone, Name, Name_reseau)
            # fichier = open(fichier_list[ind[0]], 'a')
            # fichier.write(str(couple.loc[i][0]) + ' ' + str(couple.loc[i][1]) + ' '
            #               + str(couple.loc[i][2]) + ' ' + str(couple.loc[i][3]) + ' ' + str(couple.loc[i][4]) + '\n')
            # fichier.close()
    couple = couple[couple[:, 4] == 1]
    '''if len(couple) != 0:
        couple = run_filter(couple)
        print(couple)'''
    return couple


def final_train_sans_dask(pg, poly_geopandas, nuage ):
    """

    :param pg:point cle  chambre
    :param poly_geopandas:chambre en format polygone
    :param nuage: les nuages de points en general
    :return: list de tout les chambres bien determiner d'un tel reseau
    """
    CHMBR = []
    for i in range(len(pg)):#116#nbr de chambre
        data_final = cherche_dask(i, nuage, pg)
        # data_final = data_final.reset_index(drop = True)
        CHMBR.append(definitiv_chambre(data_final, poly_geopandas))

    return CHMBR


def threads_process(pg, poly_geopandas, nuage):
    """

    :param data: list de tout les polylignes
    :param nuage: tout les pointcloud
    :param marge: contour pour tout les sommet
    :return: list difference entre nuage et sommet
    """
    # split data pour chaque processeur
    decouper = []
    df = []
    # shape_init = 3
    shape_init = int(len(pg) / 3)
    decouper.append(pg[:shape_init])
    # threads:1
    decouper.append(pg[shape_init:shape_init*2])
    # threads:2
    decouper.append(pg[shape_init*2:])
    # threads:3
    for v in decouper:
        df.append(delayed(final_train_sans_dask)(v, poly_geopandas, nuage ))
    ddf = dask.compute(*df)
    return ddf


# ---------------------------filtrage de chambre avec z_elevation---------------------- #


def initialise_stat(fil_list, Z_eleva):
    """

    :param fil_list:list de tout les chambre (numpy array)
    :param Z_eleva: list de z_elevation de chaque polyligne
    :return:
    """
    stat_descr = pd.DataFrame()
    Min_chambr = []
    Max_chambr = []

    for i in range(len(fil_list)):
        if len(fil_list[i]) != 0:
            Min_chambr.append(min(fil_list[i][:, 2]))
            Max_chambr.append(max(fil_list[i][:, 2]))
        else :
            Min_chambr.append(np.nan)
            Max_chambr.append(np.nan)
    stat_descr['Min/chambr'] = Min_chambr
    stat_descr['Z_elevation'] = Z_eleva
    stat_descr['Max/chambr'] = Max_chambr
    stat_descr['Diff_MIn'] = stat_descr['Min/chambr'] - stat_descr['Z_elevation']
    stat_descr['Diff_Max'] = stat_descr['Max/chambr'] - stat_descr['Z_elevation']
    return stat_descr


def get_filtre(stat_descr):
    stat_descr['Z_filter_Mean'] = stat_descr.Z_elevation + np.nanmedian(np.array(stat_descr.Diff_MIn))
    alpha_coeff = stat_descr[stat_descr['Max/chambr'] == np.nanmin(np.array(stat_descr['Max/chambr']))]
    if len(alpha_coeff) != 0 :
        alpha_index = alpha_coeff.index[0]
        alpha_coeff = alpha_coeff.Diff_Max[alpha_index]
        stat_descr['Z_filter_Max_Min'] = stat_descr.Z_elevation + alpha_coeff # Utilisation Max_Min
    else :
        stat_descr['Z_filter_Max_Min'] = stat_descr.Z_elevation
    return stat_descr


def Isolation(stat_descr, file_list):
    """
    effacer tout les chambres qui  a de charge anormale
    :param stat_descr:
    :param file_list: list de tout les chambre
    :return: chambre filtre
    """
    CHMBR_FILTRE = []
    for i in range(len(stat_descr)):
        # chmbr = pd.read_csv(stat_descr.Name_chambre.loc[i], header=None, sep=' ')
        chmbr = file_list[i]
        if len(chmbr):
            # tester le z du chambre est inferieur au z moyen
            chmbr = chmbr[chmbr[:, 2] <= stat_descr.Z_filter_Mean[i] + 0.3]
        if len(chmbr) >= 2:
            CHMBR_FILTRE.append(chmbr)
    return CHMBR_FILTRE


def filtre_chmbre(data_stack):
    """

    :param data_stack:numpy dataframe
    :return:numpy dataframe
    """
    for i in range(len(data_stack)):
        if len(data_stack[i]) != 0:
            if max(data_stack[i][:,2]) - min(data_stack[i][:,2]) >= 1 :
                model = KMeans(2)
                model.fit(np.array(data_stack[i][:,2]).reshape((-1,1)))
                label = model.labels_
                df = pd.DataFrame(data_stack[i])
                df['label'] = label
                df_0 = df[df['label'] == 0]
                df_1 = df[df['label'] == 1]
                mean_0 = df_0[2].mean()
                mean_1 = df_1[2].mean()
                df_0 = df_0.drop('label', axis=1)
                df_1 = df_1.drop('label', axis=1)
                if mean_0 <= mean_1:
                    data_stack[i] = np.array(df_0)
                else:
                    data_stack[i] = np.array(df_1)
    return data_stack


def run_stat(dxf_file, nuage):
    pg, poly_geopandas, Z_eleva = dxf_to_geopandas(dxf_file)
    list_chambr = threads_process(pg, poly_geopandas, nuage)
    data_stack = []
    for i in list_chambr[0]:
        data_stack.append(i)
    for k in list_chambr[1]:
        data_stack.append(k)
    for h in list_chambr[2]:
        data_stack.append(h)

    data_stack = filtre_chmbre(data_stack)
    # stat_descr = initialise_stat(data_stack, Z_eleva)
    # stat_descr = get_filtre(stat_descr)
    # return Isolation(stat_descr, data_stack)
    return data_stack


# ---------------------------Down sample chaque chambre ----------------------------------------- #


def get_max_min_chmbre(chambr_filter):
    """

    :param chambr_filter:list chambre
    :return: list max_min chambre
    """
    chmbr_finl = []
    for i in range(len(chambr_filter)):
        cmbr = chambr_filter[i]
        if len(cmbr) != 0:
            chmbr_finl.append([min(cmbr[:, 0]), max(cmbr[:, 0]),
                               min(cmbr[:, 1]), max(cmbr[:, 1])])#min_x,max_x,min_y,max_y

    return chmbr_finl


def encadrage_chambr(nuage, marge, cmbr_cadre):
    """

    :param nuage:tous les nuages de points
    :param marge:contour de chaque chambre
    :param cmbr_cadre: max_min de chaque chambre
    :return: chambre entoure avec de non chambre
    """
    x_temp = []
    for j in range(len(cmbr_cadre)):
        dfj = nuage[(nuage[:, 0] >= (cmbr_cadre[j][0] - marge))
                    & (nuage[:, 0] <= (cmbr_cadre[j][1] + marge))
                    & (nuage[:, 1] >= (cmbr_cadre[j][2] - marge))
                    & (nuage[:, 1] <= (cmbr_cadre[j][3] + marge))]
        # dfj = run_filter(dfj)
        x_temp.append(dfj)
    return x_temp


# ---------------------------Labelisation de chaque points ----------------------------------------------- #


def get_total_chambre(chmbr_filter):
    if len(chmbr_filter) != 0:
        return np.concatenate([i for i in chmbr_filter], axis=0)
    else:
        return chmbr_filter


def process_intensity(array):
    """
    translation de l'intensite vers le plage (13120.0 ; 48377.0 )
    :param array:
    :return:
    """
    if len(array)!=0:
        print('*' * 100)
        # data = pd.read_csv('D:\\VR3D\code\\ML-VR3D\\code\\ML-VR3D\\Pretraitement\\data\\out2.txt')
        # array = np.array(data)
        array = array[(array[:, 3] >= np.quantile(array[:, 3], 0.1)) &
                      (array[:, 3] <= np.quantile(array[:, 3], 0.99))]
        min_arr = min(array[:, 3])
        max_arr = max(array[:, 3])

        # 13120.0 = a * min_arr + b
        # 48377.0 = a * max_arr + b
        a = (13120.0 - 48377.0)/(min_arr - max_arr)
        b = 13120.0 - a * min_arr
        array[:, 3] = array[:, 3] * a + b
        print('stat describe', stats.describe(array[:, 3]))
    return array


def training_fichier(input_data, daska_data):
    """

    :param input_data:nuage_marge
    :param daska_data:chambre filter
    :return:data labelize
    """
    if len(daska_data)!=0:
        if len(input_data)!=0:
            input_data = np.concatenate([i for i in input_data], axis=0)
            ddf = pd.DataFrame(input_data)
            daska_data = pd.DataFrame(daska_data)
            if ddf.shape[0] >= 2:
                daska_data = daska_data.rename(columns={0: 'X', 1: 'Y', 2: 'Z', 3: 'I', 4: 'L'})
                ddf = ddf.rename(columns={0: 'X', 1: 'Y', 2: 'Z', 3: 'I'})
                ddf['L'] = 0

                data_labelize = pd.merge(ddf,daska_data,on=['X','Y','Z'])
                data_labelize = data_labelize.drop(columns=['I_y','L_x'])
                data_labelize = data_labelize.rename(columns={'I_x':'I','L_y':'L'})
                data_labelize = pd.concat(
                    [ddf,data_labelize],ignore_index=True).drop_duplicates(subset = ['X','Y','Z'],keep = 'last' )
                data_labelize = data_labelize.reset_index(drop = True)

            # data_labelize.to_csv('D:\VR3D\code\ML-VR3D\code\ML-VR3D\Pretraitement\data\Z1.txt', index = False)
            return data_labelize
        else :
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# ---------------------------Transformation array to h5----------------------------------------------- #

def array_to_h5(chambr_filter, output_filename_prefix):
    # data_dir = "D:\VR3D\VR3D_DATASET\Out_nuage\Filtre_intensite\Filtre_contour"
    # indoor3d_data_dir = os.path.join(data_dir, 'stanford_indoor3d')

    # NUM_POINT = 1024

    # Set paths
    # filelist = os.path.join(data_dir, 'my_data_label.txt')
    # = [os.path.join(data_dir, line.rstrip()) for line in open(filelist)]


    # output_dir = os.path.join(data_dir, 'data\h5_data')

    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)

    # output_filename_prefix = os.path.join(output_dir, 'train_VR3D_18_02_2022_H5')

    # output_filename_prefix = os.path.join(output_dir, 'test_eclated_from_H5_100')

    # output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
    # fout_room = open(output_room_filelist, 'w')
    # Write numpy array data and label to h5_filename
    def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
        h5_fout = h5py.File(h5_filename, 'w')
        h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
        h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
        h5_fout.close()


    def room2blocks_randomed_wrapper_normalized(data_label, num_point, block_size=1.0, stride=1.0,
                                                random_sample=False, sample_num=None, sample_aug=1):
        '''if data_label_filename[-3:] == 'txt':
            # data_label = np.loadtxt(data_label_filename)
            data_label = pd.read_csv(data_label_filename).to_numpy()
        elif data_label_filename[-3:] == 'npy':
            data_label = np.load(data_label_filename)
        elif data_label_filename[-3:] == 'uet':
            data_label = pd.read_parquet(data_label_filename).to_numpy()
        else:
            print('Unknown file type! exiting.')
            exit()'''
        return room2blocks_randomed_plus_normalized(data_label, num_point, block_size, stride,
                                                    random_sample, sample_num, sample_aug)

    def room2blocks_randomed_plus_normalized(data_label, num_point, block_size, stride,
                                             random_sample, sample_num, sample_aug):
        """ room2block, with input filename and RGB preprocessing.
            for each block centralize XYZ, add normalized XYZ as 678 channels
        """
        print('\nmise en block...')
        data = data_label[:, 0:4]
        # data[:,3:6] /= 255.0
        label = data_label[:, -1].astype(np.uint8)
        max_room_x = max(data[:, 0])
        max_room_y = max(data[:, 1])
        max_room_z = max(data[:, 2])

        # data_batch, label_batch = room2blocks_random(data_label, num_point)
        data_batch, label_batch = room2blocks_no_random(data_label, num_point)

        new_data_batch = np.zeros((data_batch.shape[0], num_point, 7))
        for b in range(data_batch.shape[0]):
            new_data_batch[b, :, 4] = data_batch[b, :, 0] / max_room_x
            new_data_batch[b, :, 5] = data_batch[b, :, 1] / max_room_y
            new_data_batch[b, :, 6] = data_batch[b, :, 2] / max_room_z
            minx = min(data_batch[b, :, 0])
            miny = min(data_batch[b, :, 1])
            data_batch[b, :, 0] -= (minx + block_size / 2)
            data_batch[b, :, 1] -= (miny + block_size / 2)
        new_data_batch[:, :, 0:4] = data_batch

        return new_data_batch, label_batch

    def room2blocks_random(data, num_point):
        # RandomisÃ© data numpy
        np.random.shuffle(data)
        # calculer nombre de blocs de 4096 dans data
        if (data.shape[0] % num_point == 0):
            num_block = data.shape[0] / num_point
        else:
            num_block = data.shape[0] / num_point + 1

        print('num_block = ', num_block)

        # Collect blocks
        block_data_list = []
        block_label_list = []
        idx = 0

        for i in range(int(num_block)):
            # Recuper donnees x,y,z,i
            block_data_sampled = data[int(num_point) * i:int(num_point) * (i + 1), 0:4]

            # Recuper donnees x,y,z,i
            block_label_sampled = data[int(num_point) * i:int(num_point) * (i + 1), -1]

            # print('---------------------------------------------------')
            # print(block_data_sampled)
            # print('---------------------------------------------------')

            # block_label_list.append(np.expand_dims(block_label_sampled, 0))
            if block_data_sampled.shape[0] < num_point:
                sample = np.random.choice(block_data_sampled.shape[0], num_point - block_data_sampled.shape[0])
                dup_data = data[sample, 0:4]
                dup_label = data[sample, -1]
                block_data_sampled = np.concatenate([block_data_sampled, dup_data], 0)

                block_label_sampled = np.concatenate([block_label_sampled, dup_label], 0)
            # print(block_data_sampled.shape[0])
            # print(block_label_sampled.shape[0])

            block_data_list.append(np.expand_dims(block_data_sampled, 0))
            block_label_list.append(np.expand_dims(block_label_sampled, 0))

        return np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)


    def room2blocks_no_random(data, num_point):
        # calculer nombre de blocs de 4096 dans data
        if (data.shape[0] % num_point == 0):
            num_block = data.shape[0] / num_point
        else:
            num_block = data.shape[0] / num_point + 1

        print('num_block = ', num_block)

        # Collect blocks
        block_data_list = []
        block_label_list = []
        idx = 0

        for i in range(int(num_block)):
            # Recuper donnees x,y,z,i
            block_data_sampled = data[int(num_point) * i:int(num_point) * (i + 1), 0:4]

            # Recuper donnees x,y,z,i
            block_label_sampled = data[int(num_point) * i:int(num_point) * (i + 1), -1]

            # print('---------------------------------------------------')
            # print(block_data_sampled)
            # print('---------------------------------------------------')

            # block_label_list.append(np.expand_dims(block_label_sampled, 0))
            if block_data_sampled.shape[0] < num_point:
                sample = np.random.choice(block_data_sampled.shape[0], num_point - block_data_sampled.shape[0])
                dup_data = data[sample, 0:4]
                dup_label = data[sample, -1]
                block_data_sampled = np.concatenate([block_data_sampled, dup_data], 0)
                block_label_sampled = np.concatenate([block_label_sampled, dup_label], 0)
            # print(block_data_sampled.shape[0])
            # print(block_label_sampled.shape[0])

            block_data_list.append(np.expand_dims(block_data_sampled, 0))
            block_label_list.append(np.expand_dims(block_label_sampled, 0))

        return np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)

    # --------------------------------------
    # ----- BATCH WRITE TO HDF5 -----
    # --------------------------------------


    def insert_batch(data, label, last_batch=False):
        global h5_batch_data, h5_batch_label
        global buffer_size, h5_index
        data_size = data.shape[0]
        # print("\ninsert_batch...")

        # If there is enough space, just insert
        if buffer_size + data_size <= h5_batch_data.shape[0]:
            # print('if :', buffer_size + data_size, ' <= ', h5_batch_data.shape[0])
            # print("there is space... insert new batch")
            h5_batch_data[buffer_size:buffer_size + data_size, ...] = data
            h5_batch_label[buffer_size:buffer_size + data_size] = label
            buffer_size += data_size
        else:  # not enough space
            # print('if :', buffer_size + data_size, ' > ', h5_batch_data.shape[0])
            # print("Space is not enough... ")
            capacity = h5_batch_data.shape[0] - buffer_size
            assert (capacity >= 0)
            if capacity > 0:
                h5_batch_data[buffer_size:buffer_size + capacity, ...] = data[0:capacity, ...]
                h5_batch_label[buffer_size:buffer_size + capacity, ...] = label[0:capacity, ...]
                # Save batch data and label to h5 file, reset buffer_size
            h5_filename = output_filename_prefix + '_' + str(h5_index) + '.h5'
            save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
            # print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
            h5_index += 1
            buffer_size = 0
            # recursive call
            insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
        if last_batch and buffer_size > 0:
            # print('if : last_batch = True et buffer_size = ', buffer_size, 'est >0')

            h5_filename = output_filename_prefix + '_' + str(h5_index) + '.h5'
            save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...],
                      data_dtype, label_dtype)
            # print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
            h5_index += 1
            buffer_size = 0

        print('\nFINISHED.\nNext h5_index for filename output = ', h5_index, '\n')
        return

    # output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
    # fout_room = open(output_room_filelist, 'w')


    def run_h5(file_list):
        file_name = [file_list]
        sample_cnt = 0
        for i, data_label_filename in enumerate(file_name):
            # block_size = 200000.0
            # stride = 100000.0
            # print(data_label_filename)
            data, label = room2blocks_randomed_wrapper_normalized(data_label_filename, NUM_POINT, block_size=1.0,
                                                                  stride=0.5,
                                                                  random_sample=False, sample_num=None)
            # print('\n Sortie Data shape = {0}, Label shape = {1}'.format(data.shape, label.shape))
            # for _ in range(data.shape[0]):
            #     fout_room.write(os.path.basename(data_label_filename) + '\n')

            # sample_cnt += data.shape[0]

            # print('IS i=', i, ' === ', 'len(file_name)-1=', len(file_name) - 1)
            insert_batch(data, label, i == len(file_name) - 1)

        # fout_room.close()
        # print("Total samples: {0}".format(sample_cnt))
    run_h5(chambr_filter)


def concate_h5():
    list_h5 = pd.read_csv("D:\VR3D\code\ML-VR3D\code\ML-VR3D\Pretraitement\data\h5_data\List_file.txt", header=None)
    label = h5py.File(list_h5[0][0], 'r')['label'][:]
    data = h5py.File(list_h5[0][0], 'r')['data'][:]
    for i in range(1, len(list_h5[0])):
        if len(label) < 1000:
            if len(label_i) < 1000:
                label_i = h5py.File(list_h5[0][i], 'r')['label'][:]
                data_i = h5py.File(list_h5[0][i], 'r')['data'][:]
                data = np.append(data, data_i, axis=0)
                label = np.append(label, label_i, axis=0)
            else :
                print(f"file n:{i} bien organise")


def main():
    args = parse_args()
    list_file = pd.read_csv("D:\VR3D\VR3D_DATASET\VR3D_PTS\list_file.txt", header=None)
    total = pd.DataFrame()
    for i in range(len(list_file)):
        local_var = np.array(pd.read_csv(list_file[0][i]))
        # print(pd.read_csv(list_file[0][i], sep = ' ',header=None))
        chambr_filter = run_stat(args.dxf_path, local_var)
        chmbr_cadre = get_max_min_chmbre(chambr_filter)
        nuage_marge = encadrage_chambr(local_var, args.marge, chmbr_cadre)
        nuage_marge = filtre_chmbre(nuage_marge)
        total_chambr = get_total_chambre(chambr_filter)
        total = total.append(training_fichier(nuage_marge, total_chambr), ignore_index=True)
        print('finish', i)
    # total.to_csv('D:\VR3D\code\ML-VR3D\code\ML-VR3D\Pretraitement\data\Z12.txt', index=False)
    array_final = process_intensity(np.array(total))
    pd.DataFrame(array_final).to_csv('D:\VR3D\code\ML-VR3D\code\ML-VR3D\Pretraitement\data\Z11_rect_pts.txt', index=False)
    array_to_h5(array_final, args.h5_path)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print('time execution in ', t2-t1)
    print('process finish !')