# coding: utf-8
import time
import numpy as np
import argparse
import dask
import warnings
import os
import pandas as pd
import ezdxf.math as math

import ezdxf
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from detectChambre import *
from dask import delayed
from LandMarkDetectionV1 import *
from Filter_local import *
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
        # default="D:\VR3D\VR3D_DATASET\PourML\\3-5\PUY-DE-DOME_ASS_3-5.dxf",
        # default= "D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_Bordeau\DXF_file\Z1\Max_Min_chambre.txt",
        # default="D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_Paris\DXF_file\z8\ASS\Max_Min_chambre.txt",
        default="C:\\Users\\f.ramamonjisoa\\Documents\\Dossier de travail\\3-) VR3D\VR3D-P-my\\Test\\input\\PARIS_ASS_Z4.dxf",
        help="Path to list sommet lisp",
    )
    parser.add_argument(
        "--standardform_path",
        type=str,
        default="C:\\Users\\f.ramamonjisoa\\Documents\\Dossier de travail\\3-) VR3D\VR3D-P-my\\Test\\input\\standard_form.csv",
        help="Path chambre standard forme by topo",
    )
    parser.add_argument(
        "--nuage_path",
        type=str,
        # default="D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_XX\\1-11.las",
        # default="D:\VR3D\VR3D_DATASET\J35_txt_cutted\VR3D_XX\z1-11.pts",
        # default="D:\VR3D\VR3D_DATASET\PourML\\3-5\Z3-5.pts",
        # default="D:\VR3D\VR3D_DATASET\Out_nuage\Out_filtre_Paris\Z8.las",
        default="C:\\Users\\f.ramamonjisoa\\Documents\\Dossier de travail\\3-) VR3D\VR3D-P-my\\Test\\input\\Z4.pts",
        help="Path to nuage de points topo",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        # default="D:\\VR3D\\code\\ML-VR3D\VR3D\\labelisation_automatique\Test\\test\data\list_sommet8.csv",
        default="C:\\Users\\f.ramamonjisoa\\Documents\\Dossier de travail\\3-) VR3D\VR3D-P-my\\Test\\input\\list_sommet8.csv",
        help="Path to list rectified lisp",
    )
    parser.add_argument(
        "--dxf_path",
        type=str,
        default="C:\\Users\\f.ramamonjisoa\\Documents\\Dossier de travail\\3-) VR3D\VR3D-P-my\\Test\\input\\polV6.dxf",
        help="Path to dessin dwg",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        # default="D:\\finaly_train_10_vr3d-pts_bordeau_2022-05-10-06 41 36_Down_sample_bordeau_elevation_filtre.h5",
        default="C:\\Users\\f.ramamonjisoa\\Documents\\Dossier de travail\\3-) VR3D\VR3D-P-my\\Test\\input\\Trained_ep_19_vr3d-pts_bordeau_2022-04-08-13 22 28_Down_sample_bordeau_elevation_filtre.h5",
        help="Path to model en h5",
    )
    parser.add_argument(
        "--marge",
        type=float,
        default=0.5,
        help="marge de contour",
    )
    return parser.parse_args()


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
            polys1 = polys1.append(geopandas.GeoSeries(
                [Polygon(poly)]), ignore_index=True)
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
                    C = math.bulge_center(
                        polygon_data[nxt - 1], polygon_data[nxt], bulg[nxt - 1])
                    cx.append(C[0])
                    cy.append(C[1])
                    R = math.bulge_radius(
                        polygon_data[nxt - 1], polygon_data[nxt], bulg[nxt - 1])
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
            pol.append([centerX - cercleR, centerY - cercleR,
                       centerX + cercleR, centerY + cercleR])
            polys1 = polys1.append(geopandas.GeoSeries(
                [Point(centerX, centerY).buffer(cercleR)]), ignore_index=True)
    return pol, polys1, Z_eleva


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
        sommet_i = sommet[(sommet[:, 0] >= list_sommet[i][0] - marge) &
                          (sommet[:, 0] <= list_sommet[i][2] + marge) &
                          (sommet[:, 1] <= list_sommet[i][3] + marge) &
                          (sommet[:, 1] >= list_sommet[i][1] - marge)
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
                    out_save = get_chambre_pts(list_sommet, np.array(
                        pd.DataFrame(ROW)), out_save, marge)
                    if j == 10:
                        t2 = time.time()
                        print(f'execution in {(t2 - t1)/60} min')
                        break
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
        array = array[(array[:, 3] >= np.quantile(array[:, 3], 0.1)) & (
            array[:, 3] <= np.quantile(array[:, 3], 0.99))]
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
    else:
        concat_chmbr = np.concatenate(
            [i for i in las_array[las_array[:, 1] == ind_chambr][:, 0]])
        if len(concat_chmbr) != 0:
            return process_intensity(concat_chmbr)
        else:
            return []
    # scraper nuage de points alentoure sommet_i


def convex_hull(array_3):
    hull = ConvexHull(array_3, )
    best_contour_x = []
    best_contour_y = []
    plot_arr_x = []
    plot_arr_y = []

    for i, simplex in enumerate(hull.simplices):
        plot_arr_x.append(array_3[simplex, 0])
        plot_arr_y.append(array_3[simplex, 1])
        # plt.scatter(hull.points[simplex, 0], hull.points[simplex, 1])
        best_contour_x.extend(hull.points[simplex, 0])
        best_contour_y.extend(hull.points[simplex, 1])
    return best_contour_x, best_contour_y, plot_arr_x, plot_arr_y


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """

    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    #     rotations = np.vstack([
    #         np.cos(angles),
    #         -np.sin(angles),
    #         np.sin(angles),
    #         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def del_polygon(array, contour):
    """
    comparaison par rapport a la bord de x et la bord de y
    :param array:
    :param contour:
    :return:
    """
    minx_ar = min(array[:, 0])
    maxx_ar = max(array[:, 0])
    miny_ar = min(array[:, 1])
    maxy_ar = max(array[:, 1])

    minx_cont = min(contour[:, 0])
    maxx_cont = max(contour[:, 0])
    miny_cont = min(contour[:, 1])
    maxy_cont = max(contour[:, 1])
    if (minx_ar > minx_cont - 0.01) or (
            maxx_ar < maxx_cont + 0.01) or (
            miny_ar > miny_cont - 0.01) or (
            maxy_ar < maxy_cont + 0.01):
        return []
    else:
        return contour


def PolyArea(x, y):
    """
    Calcul aire du polygone
    :param x:
    :param y:
    :return:
    """
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def detect_contour(df):
    # df = np.array(pd.read_csv(path, header=None, sep=' '))
    size_fil = len(df)
    print('size_file:', size_fil)

    if size_fil < 1000:
        eps = 0.16
    elif (size_fil > 1000) & (size_fil < 1500):
        eps = 0.14
    elif (size_fil > 1500) & (size_fil < 2000):
        eps = 0.13
    elif (size_fil > 2000) & (size_fil < 2500):
        eps = 0.12
    elif (size_fil > 2500) & (size_fil < 3000):
        eps = 0.11
    elif (size_fil > 3000) & (size_fil < 3500):
        eps = 0.10
    elif (size_fil > 3500) & (size_fil < 4000):
        eps = 0.09
    elif (size_fil > 4000) & (size_fil < 10000):
        eps = 0.08
    elif (size_fil > 10000) & (size_fil < 15000):
        eps = 0.04
    else:
        eps = 0.03
    objects = del_noise(df.copy(), eps)

    if len(objects) != 0:
        cont_x, cont_y, plot_x, plot_y = convex_hull(objects[:, :2])
        x_arr = np.array(cont_x).reshape((-1, 1))
        y_arr = np.array(cont_y).reshape((-1, 1))
        arr_2 = np.hstack((x_arr, y_arr))
        arr_2 = del_polygon(df, arr_2)
        if len(arr_2) != 0:
            rect = minimum_bounding_rectangle(arr_2)
            area = PolyArea(rect[:, 0], rect[:, 1])
            if area >= 0.01:
                arr_2 = np.array(pd.DataFrame(
                    arr_2).drop_duplicates(keep='first'))
                # objects = del_noise(objects, beta)
                # plt.scatter(df[:, 0], df[:, 1])
                plt.scatter(objects[:, 0], objects[:, 1])
                plt.scatter(cont_x, cont_y)
                # plt.plot(cont_x, cont_y)
                # plt.scatter(rect[:, 0], rect[:, 1])
                # plt.plot(arr_2[:, 0], arr_2[:, 1])
                plt.show()
                return plot_x, plot_y
            else:
                return [], []
        else:
            return [], []
    else:
        return [], []


def run_model(arg, list_sommet, marge, list_chambr, is_las):
    """
    run model pour une subdivision de list_chambre
    Detect_markings, separation_marquage, non_marquage
    :param arg: ref model
    :param list_chambre: subdivision chambre a traite
    :return: list chambre marque
    """
    sommet_total_x = []
    sommet_total_y = []

    for i in range(len(list_sommet)):
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                # get chambre sans transformation

                chambre = get_chambre(
                    list_chambr, i, is_las, list_sommet, marge)
                # np.savetxt(path + str(i) + '_chambre.txt', chambre, delimiter=',')

                print(np.array(chambre))

                if len(chambre) != 0:
                    chambre = run_filter(np.array(chambre))

                    # -------------------------Detection ------------------------------------------------ #
                    score_label = l1.detect_markings(
                        arg.model_path,
                        chambre.copy(),
                    )
                    # -------------------------Separation marquage non marquage     --------------------- #
                    object_Ndetect, object_detect = l1.visualize_result(
                        chambre.copy(), score_label)

                    # -------------------------Post Processing ------------------------------------------ #
                    plot_x, plot_y = detect_contour(np.array(object_Ndetect))

                    sommet_total_x.extend(plot_x)
                    sommet_total_y.extend(plot_y)
    return sommet_total_x, sommet_total_y


def stack_threads(ddf):
    sommet_x = np.concatenate([ddf[i][0] for i in range(len(ddf))], axis=None)
    sommet_y = np.concatenate([ddf[i][1] for i in range(len(ddf))], axis=None)
    return sommet_x, sommet_y


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
    print(ddf)
    sommet_x, sommet_y = stack_threads(np.array(ddf))
    return sommet_x, sommet_y


def create_outp_csv(sommet_total, center_total, result_path):
    list_sommet = []
    list_center = []
    out = pd.DataFrame()
    for sommet_list in sommet_total:
        list_sommet.extend(
            ['/'.join([';'.join([str(j) for j in sommet]) for sommet in sommet_list])])
    for center in center_total:
        list_center.extend([';'.join([str(j) for j in center])])
    if len(list_sommet) == 0:
        list_sommet = [np.nan for _ in range(len(list_center))]
    if len(list_center) == 0:
        list_center = [np.nan for _ in range(len(list_sommet))]

    if len(list_center) < len(list_sommet):
        list_center.extend([np.nan for _ in range(
            len(list_sommet) - len(list_center))])
    elif len(list_center) > len(list_sommet):
        list_sommet.extend([np.nan for _ in range(
            len(list_center) - len(list_sommet))])
    out['SOMMET'] = list_sommet
    out['CENTER'] = list_center
    out.to_csv(
        result_path,
        sep=';',
        index=False
    )


def draw_polyligne(plot_x, plot_y, dxf_path):
    """
    pour demo uniquement
    :return:
    """
    # Create a new DXF R2010 drawing, official DXF version name: "AC1024"
    doc = ezdxf.new('R12')  # create a new DXF drawing in R2010 fromat
    msp = doc.modelspace()  # add new entities to the modelspace
    """lines = [(0, 0), (0, 10),(20,10),(20, 0)]
    lines = [[1709137.99177, 5174070.9],
                  [1709136.20277, 5174070.9],
                  [1709136.20277, 5174071.4],
                  [1709137.99177, 5174071.4]]"""
    dxfattribs = {"color": 1}
    # --------------- creation polygon ----------------------------------- #
    if len(plot_x) != 0:
        # rect = sommet.copy()
        # for i in range(len(sommet)):
        #     rect.append(np.append(sommet[i]))
        for i in range(len(plot_x)):
            start = [plot_x[i][0], plot_y[i][0]]
            end = [plot_x[i][1], plot_y[i][1]]
            start.extend([62])
            end.extend([62])
            msp.add_line(start, end, dxfattribs=dxfattribs)
    # -------------- creation cercle ------------------------------------- #
    """if len(center_list) != 0:
        for center in center_list:
            if len(center) != 0:
                center.extend([62])
                msp.add_circle(
                    center=center, radius=0.3, dxfattribs={"color": 3}
                )"""
    doc.saveas(dxf_path)


def main():
    arg = parse_args()
    is_las = False
    out_save = []
    # ----------- simulation de list sommet pour demo ------------------------------------------------- #
    list_sommet, _, _ = dxf_to_geopandas(arg.list_path)
    list_sommet = np.array(list_sommet)
    # list_sommet = np.array(pd.read_csv(arg.list_path, header= None))
    # -----------------------load file las & Filtrage_Z en general ------------------------------------ #
    if arg.nuage_path.split()[-1][-3:] == 'las':
        is_las = True
        las_array = run_filter(arg.nuage_path)
        list_som_x, list_som_y = run_model(
            arg, list_sommet, arg.marge, las_array, is_las)
        # sommet_list, center_list = threads_process(arg, list_sommet, arg.marge, las_array, is_las)

    else:
        # -------------------------------- load file pts & transf to array -------------------------------- #
        list_chambr = pts_txt(list_sommet, arg.nuage_path, out_save, arg.marge)
        # ----------------------------------- separe le processus d'inference en 3 threads ---------------- #
        # list_som_x, list_som_y = threads_process(arg, list_sommet, arg.marge, list_chambr, is_las)
        list_som_x, list_som_y = run_model(
            arg, list_sommet, arg.marge, list_chambr, is_las)
    # pour demo
    draw_polyligne(list_som_x, list_som_y, arg.dxf_path)
    # create_outp_csv(sommet_list, center_list, arg.result_path)


if __name__ == '__main__':
    time_init = time.time()
    main()
    """
    list_chambr = pts_txt(list_sommet, arg.nuage_path, out_save, arg.marge)
    # ----------------------------------- separe le processus d'inference en 3 threads ---------------- #
    # list_som_x, list_som_y = threads_process(arg, list_sommet, arg.marge, list_chambr, is_las)
    list_som_x, list_som_y = run_model(
        arg, list_sommet, arg.marge, list_chambr, is_las)
    # pour demo
    draw_polyligne(list_som_x, list_som_y, arg.dxf_path)
    # create_outp_csv(sommet_list, center_list, arg.result_path)
    """
"""
if __name__ == '__main__':
    time_init = time.time()
    main()
"""
