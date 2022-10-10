# coding: utf-8
import time
import numpy as np
import argparse
import dask
import warnings
import pandas as pd

import ezdxf
import matplotlib.pyplot as plt
from detectChambre import *
from dask import delayed
from LandMarkDetectionV1 import *
from Filter_Z import *
from ezdxf.addons import odafc
import subprocess
# from ezdxf.math import Vec3
from scipy import stats

warnings.filterwarnings("ignore")
# laspy version 1.7.0

l1 = LandMarkDetection()
# list model a visualiser
parameters = {
    'refmod': [
        'models'
    ]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Detection chambre ")
    parser.add_argument(
        "--list_path",
        type=str,
        default="D:\VR3D\code\ML-VR3D\VR3D\labelisation_automatique\Test\\test\data\list_point1.csv",
        # default="D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_Paris\DXF_file\z8\ASS\Max_Min_chambre.txt",
        help="Path to list sommet lisp",
    )
    parser.add_argument(
        "--standardform_path",
        type=str,
        default="D:\VR3D\code\ML-VR3D\code\ML-VR3D\Pretraitement\data\standard_form.csv",
        help="Path chambre standard forme by topo",
    )
    parser.add_argument(
        "--nuage_path",
        type=str,
        # default="D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_XX\\1-11.las",
        # default="D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_XX\z1-11.pts",
        default="D:\VR3D\VR3D_DATASET\VR3D_PTS\First_vr3d_PTS.txt",
        # default="D:\VR3D\VR3D_DATASET\Out_nuage\Out_filtre_Paris\Z8.las",
        help="Path to nuage de points topo",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="D:\\VR3D\\code\\ML-VR3D\VR3D\\labelisation_automatique\Test\\test\data\list_sommet6.csv",
        help="Path to list rectified lisp",
    )
    parser.add_argument(
        "--dxf_path",
        type=str,
        default="D:\\VR3D\\code\\ML-VR3D\VR3D\\labelisation_automatique\Test\\test\data\polyligne6.dxf",
        help="Path to dessin dwg",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="D:\VR3D\SAVE_MODEL\\best_model\Trained_ep_15_VR3D_intensity2022-02-11-06 12 47_Down_sample_bordeau_elevation_filtre.h5",
        help="Path to model en h5",
    )
    parser.add_argument(
        "--marge",
        type=float,
        default=1,
        help="marge de contour",
    )
    return parser.parse_args()


def get_chambre_pts(list_sommet, sommet, out_save, marge):
    """
    Scrape tout les chambre encadre correspond a chaque
    element du list pour une subdivision du nuage total
    :param list_sommet: list de tout les sommet scraper par le topo
    :param sommet: subdivision du nuage total
    :param out_save: Dataframe a stocke du chambre
    :param marge:
    :return: list chambre cadre
    """
    for i in range(len(list_sommet)):
        sommet_i = sommet[(sommet[:, 0] >= list_sommet[i][0] + 0.5 - marge) &
               (sommet[:, 0] <= list_sommet[i][0] + 0.5 + marge) &
               (sommet[:, 1] <= list_sommet[i][1] + 0.5 + marge) &
               (sommet[:, 1] >= list_sommet[i][1] + 0.5 - marge)
        ]
        out_save.append([sommet_i, i])
        # print(f'chambre n :{i}')
    return out_save


def pts_txt(list_sommet, path, out_save, marge):
    """takes as input the path to a .pts and returns a list of
    tuples of floats containing the points in in the form:
    [(x_0, y_0, z_0),
     (x_1, y_1, z_1),
     ...
    (x_n, y_n, z_n)]"""
    ROW = []
    j = 0
    with open(path) as f:
        t1 = time.time()
        for i, row in enumerate(f):
            if i > 0:
                ROW.append(np.array(row.strip().split(), dtype="float64"))
                # ROW.append(np.array(row))
                if i % 10000000 == 0:
                    print('<&&&&')
                    j += 1
                    out_save = get_chambre_pts(list_sommet, np.array(pd.DataFrame(ROW)), out_save, marge)
                    """if j == 5:
                        t2 = time.time()
                        print(f'execution in {(t2 - t1)/60} min')
                        break"""
                    ROW = []
                    print(f'file n:{j} finish')

    return np.array(out_save)


def process_intensity(array):
    """
    translation de l'intensite vers le plage (13120.0 ; 48377.0 )
    :param array:
    :return:
    """
    if len(array) >= 10:
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
        # print('stat describe', stats.describe(array[:, 3]))
    return array


def get_chambre(las_array, ind_chambr, is_las, list_sommet, marge):
    """
    scrape nuage aux alentours du points dessigne sur le plan avec de 1m carre par default( pour cas du las)
    pour le pts concatene tout les sou donne du chambre n:i
    :param las_array: nuage de points filtre
    :param list_path_i: couple(x,y)
    :param marge: contour choisit
    :return: chambre
    """
    if is_las:
        chambre = las_array[
            (las_array[:, 0] >= (list_sommet[ind_chambr][0] - marge)) &
            (las_array[:, 0] <= (list_sommet[ind_chambr][0] + marge)) &
            (las_array[:, 1] >= (list_sommet[ind_chambr][1] - marge)) &
            (las_array[:, 1] <= (list_sommet[ind_chambr][1] + marge))]

        return chambre
    else :
        concat_chmbr = np.concatenate([i for i in las_array[las_array[:, 1] == ind_chambr][:, 0]])
        if len(concat_chmbr) != 0:
            return process_intensity(concat_chmbr)
        else :
            return []
    # scraper nuage de points alentoure sommet_i


def run_model(arg, list_sommet, marge, list_chambr, is_las):
    """
    run model pour une subdivision de list_chambre
    Detect_markings, separation_marquage, non_marquage
    :param arg: ref model
    :param list_chambre: subdivision chambre a traite
    :return: list chambre marque
    """
    sommet_chmbr = []
    center = []
    sommet_total = []
    center_total = []
    for i in range(len(list_sommet)):
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                # get chambre sans transformation

                chambre = get_chambre(list_chambr, i, is_las, list_sommet, marge)

                # np.savetxt(path + str(i) + '_chambre.txt', chambre, delimiter=',')
                if len(chambre) != 0 :
                    # -------------------------Detection ------------------------------------------------ #
                    # score_label = l1.detect_markings(
                    #     arg.model_path,
                    #     chambre.copy(),
                    # )
                    # -------------------------Separation marquage non marquage     --------------------- #
                    # object_Ndetect, object_detect = l1.visualize_result(chambre.copy(), score_label)
                    # -------------------------Post Processing ------------------------------------------ #
                    sommet_chmbr, center = post_infer(np.array(chambre.copy()), arg.standardform_path)
                sommet_total.append(sommet_chmbr)
                center_total.append(center)
    return sommet_total, center_total


def threads_process(arg, list_sommet, marge, list_chambr, is_las):
    """

    :param data: list de tout les polylignes
    :param nuage: tout les pointcloud
    :param marge: contour pour tout les sommet
    :return: list difference entre nuage et sommet
    """
    # --------------------- split data pour chaque processeur ---------------------------------------- #
    decouper = []
    df = []
    # shape_init = 3
    shape_init = int(len(list_sommet) / 3)
    decouper.append(list_sommet[:shape_init, ])
    # threads:1
    decouper.append(list_sommet[shape_init:shape_init*2, ])
    # threads:2
    decouper.append(list_sommet[shape_init*2:, ])
    # threads:3
    # --------------------- split data pour chaque processeur ---------------------------------------- #
    for v in decouper:
        df.append(delayed(run_model)(arg, v, marge, list_chambr, is_las))

    # --------------------- fit les resultat du processus ensemble------------------------------------ #
    ddf = dask.compute(*df)
    sommet_list, center_list = stack_threads(np.array(ddf))
    return sommet_list, center_list


def stack_threads(ddf):
    sommet = np.concatenate([ddf[i][0] for i in range(len(ddf))], axis=None)
    center = np.concatenate([ddf[i][1] for i in range(len(ddf))], axis=None)
    """print('apres stack ............................')
    print(len(sommet))
    print(len(center))
    print(sommet)
    print(center)"""
    return  sommet, center


def create_outp_csv(sommet_total, center_total, result_path):
    list_sommet = []
    list_center = []
    out = pd.DataFrame()
    for sommet_list in sommet_total:
        list_sommet.extend(['/'.join([';'.join([str(j) for j in sommet]) for sommet in sommet_list])])
    for center in center_total:
        list_center.extend([';'.join([str(j) for j in center])])
    if len(list_sommet) == 0:
        list_sommet = [np.nan for _ in  range(len(list_center))]
    if len(list_center) == 0:
        list_center = [np.nan for _ in  range(len(list_sommet))]

    if len(list_center) < len(list_sommet):
        list_center.extend([np.nan for _ in range(len(list_sommet) - len(list_center))])
    elif len(list_center) > len(list_sommet):
        list_sommet.extend([np.nan for _ in range(len(list_center) - len(list_sommet))])
    out['SOMMET'] = list_sommet
    out['CENTER'] = list_center
    out.to_csv(
        result_path,
        sep=';',
        index=False
    )


def draw_polyligne(sommet_list, center_list, dxf_path):
    """
    pour demo uniquement
    :return:
    """
    # Create a new DXF R2010 drawing, official DXF version name: "AC1024"
    doc = ezdxf.new('R12') # create a new DXF drawing in R2010 fromat
    msp = doc.modelspace()  # add new entities to the modelspace
    """lines = [(0, 0), (0, 10),(20,10),(20, 0)]
    lines = [[1709137.99177, 5174070.9],
                  [1709136.20277, 5174070.9],
                  [1709136.20277, 5174071.4],
                  [1709137.99177, 5174071.4]]"""
    dxfattribs = {"color": 1}
    # --------------- creation polygon ----------------------------------- #
    if len(sommet_list) != 0:
        for sommet in sommet_list:
            if len(sommet) != 0:
                rect  = []
                for i in range(4):
                    rect.append(np.append(sommet[i],[62], axis=0))
                for i in range(4):
                    start = rect[i-3]
                    end = rect[i-2]
                    msp.add_line(start, end, dxfattribs = dxfattribs)
    # -------------- creation cercle ------------------------------------- #
    if len(center_list) != 0:
        for center in center_list:
            if len(center) != 0:
                center.extend([62])
                msp.add_circle(
                    center=center, radius=0.3, dxfattribs={"color": 3}
                )
    doc.saveas(dxf_path)

    # dwg_path = 'D:\polyligne26.dwg'
    # odafc.export_dwg(doc, dwg_path)


def main():
    arg = parse_args()
    is_las = False
    out_save = []
    # ------------------------------------load list sommet -------------------------------------------- #
    # list_sommet = np.loadtxt(arg.list_path, delimiter=';', skiprows=1)
    # list_path = "D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_XX\DXF_file\Z1\Max_Min_chambre.txt"
    list_path = "D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_Bordeau\DXF_file\Z1\Max_Min_chambre.txt"
    # list_sommet = np.loadtxt(list_sommet, delimiter=';', skiprows=1)
    list_sommet = np.array(pd.read_csv(list_path, header= None))
    # -----------------------load file las & Filtrage_Z en general ------------------------------------ #
    if arg.nuage_path.split()[-1][-3:] == 'las':
        is_las = True
        las_array = run_filter(arg.nuage_path)
        # sommet_list, center_list = run_model(arg, list_sommet, arg.marge, las_array, is_las)
        sommet_list, center_list = threads_process(arg, list_sommet, arg.marge, las_array, is_las)

    else :
        # -------------------------------- load file pts & transf to array -------------------------------- #
        list_chambr = pts_txt(list_sommet, arg.nuage_path, out_save, arg.marge)
        # ----------------------------------- separe le processus d'inference en 3 threads ---------------- #
        sommet_list, center_list = threads_process(arg, list_sommet, arg.marge, list_chambr, is_las)
        # sommet_list, center_list = run_model(arg, list_sommet, arg.marge, list_chambr, is_las)
    # pour demo
    draw_polyligne(sommet_list, center_list, arg.dxf_path)
    create_outp_csv(sommet_list, center_list, arg.result_path)


if __name__ == '__main__':
    time_init = time.time()
    main()
    # draw_polyligne()
    """a = [[[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [1709262.7, 5174067.899999999], [], [], [], []]],
        [[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [1709262.7, 5174067.899999999], [], [1709247.0, 5174077.8], [], []]],
        [[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [[1709138.02977, 5174071.],
             [1709136.33877, 5174071.     ],
             [1709136.33877, 5174071.5    ],
             [1709138.02977, 5174071.5    ]]],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [1709262.7, 5174067.899999999], [], [1709247.1, 5174077.8], [], [], [], []]]]
    x,y = stack_threads(a)
    draw_polyligne(x,y,"D:\polc3 .dxf")
    print('execution_time in :', time.time() - time_init)
    # create_outp_csv(x,y,"D:\\VR3D\\code\\ML-VR3D\VR3D\\labelisation_automatique\Test\\test\data\list_sommet3.csv")

    print(np.array(a)[0])
    a = [[[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
      [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]],
     [[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
      [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]],
     [[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
           [[1709137.99177, 5174070.9],
                  [1709136.20277, 5174070.9],
                  [1709136.20277, 5174071.4],
                  [1709137.99177, 5174071.4]]],
    [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]]]
    # print(np.array(np.concatenate([a[i][0] for i in range(len(a))], axis=None)))
    sommet_list, center_list = stack_threads(a)
    D:\VR3D\code\ML-VR3D\VR3D\labelisation_automatique\Test\test\data\list_sommet3.csv
    create_outp_csv(sommet_list, center_list, "D:\\VR3D\\code\\ML-VR3D\VR3D\\labelisation_automatique\Test\\test\data\list_sommet.csv")"""