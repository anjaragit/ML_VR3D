import pandas as pd
import numpy as np
import os
import sys
import dask
import time
import argparse
from dask import delayed
from laspy.file import File
from filterZ import *
import csv
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument(
        "--data_path",
        type=str,
        default="D:\ML_Polyligne_Detection\data\data.csv",
        help="Path to data lisp",
    )
    parser.add_argument(
        "--nuage_path",
        type=str,
        default="D:\ML_Polyligne_Detection\data\Z3.las",
        help="Path to nuage de points topo",
    )

    parser.add_argument(
        "--list_path",
        type=str,
        default="D:\ML_Polyligne_Detection\data\list_charge3.csv",
        help="Path to nuage de points topo",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="D:\ML_Polyligne_Detection\data\data_rect_V23.csv",
        help="Path to data rectified lisp",
    )
    parser.add_argument(
        "--marge",
        type=float,
        default=0.5,
        help="marge de contour pour chaque sommet",
    )
    return parser.parse_args()


def xyz_sommet(data, ind):
    """

    :param data:list polyligne csv
    :param ind: ind de polyligne a travailler
    :return: coordonne (x,y,z) de chaque sommet
    """
    # flatten tout les sommet dans des dataframe xy_array et z_array
    array = np.array([line.split(';')
                     for line in data[data.columns[-1]].values[ind].split('/')])
    xy_array = pd.DataFrame(
        {'X': pd.to_numeric(pd.DataFrame(array[:, 0])[0]), 'Y': pd.to_numeric(pd.DataFrame(array[:, 1])[0])})
    z_array = pd.DataFrame({'Z': pd.to_numeric(pd.DataFrame(array[:, 2])[0])})
    return xy_array, z_array


def scrap_nuage(ddf, marge, xy_array, z_array):
    """

    :param ddf: nuage
    :param marge: contour chambre
    :param xy_array: list coordonne de xy
    :param z_array: list coordonne de z
    :return: list z diff pour chaque sommet
    """
    dfZ = pd.DataFrame()
    # liste tout les sommet du polyligne _k
    for j in range(len(xy_array)):
        # ddf = ddf.rename(columns={0: 'X', 1: 'Y', 2: 'Z', 3: 'I'})

        # scraper nuage de points alentoure sommet_i
        dfj = ddf[(ddf[:, 0] >= (xy_array['X'][j] - marge)) & (ddf[:, 0] <= (xy_array['X'][j] + marge)) &
                  (ddf[:, 1] >= (xy_array['Y'][j] - marge)) & (ddf[:, 1] <= (xy_array['Y'][j] + marge))]

        z_diff = dfj[:, 2] - z_array['Z'][j]
        z_diff_mean = z_diff.mean()
        dfZ = dfZ.append(pd.DataFrame([z_diff_mean]), ignore_index=True)
    return dfZ


def run(data, nuage, marge):
    """

    :param data:list polyligne
    :param nuage:pointcloud
    :param marge: contour utiliser pour tout les sommet
    :return: diff_z
    """
    z_temp = []  # memorise tous les diff_z a sommet superieur a deux

    for i in range(len(data)):
        xy_array, z_array = xyz_sommet(data, i)
        dfZ = scrap_nuage(nuage, marge, xy_array, z_array)
        z_temp.append(dfZ[0].values)
    return z_temp


def threads_process(data, nuage, marge):
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
    shape_init = int(len(data) / 3)
    decouper.append(data.iloc[:shape_init])
    # threads:1
    decouper.append(data.iloc[shape_init:shape_init*2])
    # threads:2
    decouper.append(data.iloc[shape_init*2:])
    # threads:3
    for v in decouper:
        df.append(delayed(run)(v, nuage, marge))
    ddf = dask.compute(*df)
    return ddf


def stack_data(ddf):
    """

    :param ddf: list diff_z non stack
    :return: z_temp list diff_z stack sommet, data_stack list diff_z stack par polyligne
    """
    z_0 = np.concatenate([ddf[0][i] for i in range(len(ddf[0]))])
    z_1 = np.concatenate([ddf[1][i] for i in range(len(ddf[1]))])
    z_2 = np.concatenate([ddf[2][i] for i in range(len(ddf[2]))])
    z_3 = [z_0, z_1, z_2]
    data_stack = []
    for i in ddf[0]:
        data_stack.append(i)
    for k in ddf[1]:
        data_stack.append(k)
    for h in ddf[2]:
        data_stack.append(h)
    z_temp = np.concatenate([z_3[i] for i in range(3)], axis=None)
    return z_temp, data_stack


def cal_seuil(z_temp):
    """

    :param z_temp: list diff_z global
    :return: seuil_min,seuil_max
    """
    var = np.nanstd(np.array(z_temp))
    moyen = np.nanmean(np.array(z_temp))

    seuil_min = moyen - var
    seuil_max = moyen + var + 0.5
    return seuil_min, seuil_max, moyen


def detecter_anomalier_stat(seuil_min, seuil_max, dfZ):
    """
    cle F:False anomaly, T:True normal, I:Intermediaire
    :param seuil_min:
    :param seuil_max:
    :param dfZ: list diff entre z_sommet et z_nuage
    :return: cle: classe correspond a chaque sommet
    """

    cle = np.zeros((len(dfZ)), dtype=object)
    # k = 0
    for i in range(len(dfZ)):
        # detecter si le diff_z soit normal ou pas
        cle[i] = 'F'
        if pd.isna(dfZ[i]):
            cle[i] = 'I'  # -1
            # k+=1
        elif (dfZ[i] >= seuil_min) & (dfZ[i] <= seuil_max):
            cle[i] = 'T'
    # print(f'on a: {k} sommet pas de nuage trouve')
    return cle


def create_xyz(data, ddf, predict, result_path):
    """

    :param data: list de tout les polyligne
    :param ddf: list de tout les diff_z
    :param predict: cle resultat du detecter anomaly
    :param result_path:
    :return:
    """
    data_rect = data.copy()
    start = 0
    PREDICT = []
    for i, array in enumerate(ddf):
        PRED = [str(j) for j in predict[start:len(array)+start]]
        PRED = '/'.join(PRED)
        start = len(array) + start
        PREDICT.append(PRED)
    return PREDICT
    # data_rect['CLASSE_PREDICT'] = PREDICT
    # data_rect.to_csv(result_path, sep=';', index=False)


def concat_charge(ddf):
    TEMP = []
    for i in range(len(ddf)):
        for j in range(len(ddf[i])):
            TEMP.append(ddf[i][j])
    return TEMP


def correct_xyz(data, ddf, predict, mean_glob, result_path, list_path):
    """
        place le correct place du point aux alentour du moyenne
        :param data: list de tout les polyligne
        :param ddf: list de tout les diff_z
        :param predict: cle resultat du detecter anomaly
        :param result_path:
        :return:
        """
    data_rect = data.copy()
    start = 0
    PREDICT = []
    XYZ = []
    DDF = []
    TEMP = []
    for i, arr in enumerate(ddf):
        array = np.array(
            [line.split(';') for line in data[data.columns[-1]].values[i].split('/')])
        xy_array = pd.DataFrame(
            {'X': pd.to_numeric(
                pd.DataFrame(array[:, 0])[0]), 'Y': pd.to_numeric(pd.DataFrame(array[:, 1])[0])})
        z_array = pd.DataFrame(
            {'Z': pd.to_numeric(pd.DataFrame(array[:, 2])[0])})
        # print('avant_modife', z_array)
        for h, k in enumerate(predict[start:len(z_array) + start]):
            if k == 'F':
                if pd.isna(ddf[i][h]):
                    pass
                else:
                    z_array['Z'][h] += (ddf[i][h] - mean_glob)

        # print('xy_array',xy_array)
        xyz = np.concatenate((xy_array, z_array), axis=1)
        TEMP.extend(xyz)
        # xyz = pd.concat([xy_array,z_array], axis=1)
        # print(xyz)
        XYZ.append('/'.join(k for k in [';'.join(
            [str(i) for i in xyz[j]]) for j in range(len(xyz))]))

        PRED = [str(j) for j in predict[start:len(arr) + start]]
        PRED = '/'.join(PRED)
        start = len(arr) + start
        PREDICT.append(PRED)

        '''df = [str(j) for j in ddf[i]]
        df = '/'.join(df)
        DDF.append(df)'''
    charge = concat_charge(ddf)
    data_charge = pd.DataFrame(np.array(TEMP).reshape(
        (-1, 3)), columns=['X', 'Y', 'Z'])
    data_charge['charge'] = charge
    data_charge.to_csv(list_path, sep=';', index=False)
    # pd.DataFrame(DDF, columns=['LIST_CHARGE']).to_csv(list_path, index=False)
    data_rect['CLASSE_PREDICT'] = PREDICT
    data_rect['SOMMET_CORRIGER'] = XYZ
    data_rect.to_csv(result_path, sep=';', index=False)


def main():
    args = parse_args()
    data = pd.read_csv(args.data_path, encoding='cp1252', sep=';')
    # data = data.iloc[:10]
    nuage = run_filter(args.nuage_path)
    print('filtrage finish')
    ddf = threads_process(data, nuage, args.marge)
    z_temp, data_stack = stack_data(ddf)
    # dd = np.concatenate((ddf[0],ddf[1]),axis=0)
    seuil_min, seuil_max, mean_glob = cal_seuil(z_temp)
    predict = detecter_anomalier_stat(seuil_min, seuil_max, z_temp)
    # PREDICT = create_xyz(data, data_stack, predict, args.result_path)
    correct_xyz(data, data_stack, predict, mean_glob,
                args.result_path, args.list_path)
    # create_xyz(data, data_stack, predict, args.result_path)


if __name__ == '__main__':
    main()
