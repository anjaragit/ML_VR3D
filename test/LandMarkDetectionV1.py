
# -*- coding: Utf-8 -*-
import os
import sys

import datetime
import numpy as np
import tensorflow as tf
import time
from pre_processing_inference_V1 import *
from indoor3d_util_V1 import *
from models.pointnet_seg import *
from post_processing_func_V1 import *
from Filter_local import *
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)


class LandMarkDetection:
    def __init__(self, is_training=False, batch_size=1, num_point=4096, num_class=2):  # NUM_CLASSES = 2
        self.is_training = is_training
        self.batch_size = batch_size
        self.num_point = num_point
        self.num_class = num_class
        self.date_i = datetime.datetime.now()
        self.date = str(self.date_i.day) + "_" + \
            str(self.date_i.month) + "_" + str(self.date_i.year)

    @staticmethod
    def detect_markings(model_path, chambre_file, num_point=4096):
        """
        Fonction de normalisation et de filtrage
        :param model_path:path model .h5
        :param chambre_file: chambre avec filtrage global
        :param num_point:
        :return:predict score par batch
        """

        # ------------------------RUN FILTRE LOCAL & EQUILIBRE INTENSITY----------------------- #
        # las_data = chambre_file.copy()
        las_data = chambre_file.copy()
        # las_data = run_filter(chmbr)

        NUM_CLASS = 2
        batch_size = 1
        # DEFINITION GRAPH DU MODELE
        with tf.device('/cpu:0'):

            model = get_model((None, 7), NUM_CLASS)
            # model.summary()
            model.load_weights(model_path)
            s_, time, score_label = LandMarkDetection.detection_inference(model, las_data,
                                                                          batch_size, num_point)

            # fout_out_filelist = open(os.path.join(result_path, 'output_filelist.txt'), 'w')
            # print('session created..')
            '''
            for each_las_file in point_cloud_list:
                # recupère nom du fichier
                elements = each_las_file.split('/')
                filename = elements[-1]

                elements2 = filename.split('.')
                filename = elements2[0]
                project_path = os.path.join(result_path, filename)
                if not os.path.exists(project_path):
                    os.mkdir(project_path)
                out_score_label_filename = filename + '_pred'
                out_score_label_filename = os.path.join(project_path, out_score_label_filename)
                print('#####->', out_score_label_filename)

                s_, time = LandMarkDetection.detection_inference(model, each_las_file, out_score_label_filename,
                                                                 batch_size, num_point)

                print('Inference Done in - ' + str(time) + '\n\n')
                fout_out_filelist.write(project_path + '\n')

            fout_out_filelist.close()'''
        return score_label

    @staticmethod
    def detection_inference(ops, each_las_file, batch_size, num_point):
        """

        :param ops: model
        :param each_las_file: nuage de points pour un chambre a inference
        :param batch_size:
        :param num_point:
        :return: g_score_label: score par batch model
        """
        td = time.time()
        is_training = False
        current_data, seed = numpy2blocks_normalized(
            each_las_file, num_point, shuffle=False)
        current_data = current_data[:, 0:num_point, :]

        num_batches = current_data.shape[0]
        all_score_label = []

        # batch par rapport taille du fichier
        for batch_idx in range(num_batches):
            # print(batch_idx)
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            cur_batch_size = end_idx - start_idx

            data_to_predict = current_data[start_idx:end_idx, :, :]

            pred_val = ops.predict(data_to_predict)
            # pred_val = sess.run(ops.predict(data_to_predict))
            pred_label = np.argmax(pred_val, axis=2)  # BxN
            # pred_label = np.argmax(pred_val[0,:])

            # Eliminer les colonnes à une dimensions
            pred_val = np.squeeze(pred_val)  # shape: (NUM_POINT * 2)
            pred_label = np.squeeze(pred_label)  # shape: (NUM_POINT)

            # SAUVEGARDER SCORE ET LABELS DANS UN FICHIERS TXT/NUMPY
            pred_val_list = []
            for i in range(num_point):
                pred_val_list.append(pred_val[i, pred_label[i]])

            pred_val_numpy = np.array(pred_val_list)
            score_label = np.column_stack((pred_val_numpy, pred_label))
            all_score_label.append(score_label)

        g_score_label = np.array(all_score_label)
        # np.save(out_score_label_filename, g_score_label)
        # print('Score de label : \n\t' + str(g_score_label),
        #       '\n a ete sauvegarder dans :\n\t' + str(out_score_label_filename) + '\n')

        timef = time.time() - td
        return seed, round((timef), 2), g_score_label

    @staticmethod
    def create_data_obj(numpy_file, pred_file):
        # Relaod numpy file and converting to blocks of batch with shape: ((nombre_points // 4096) x 4096 x 4 )
        xyzi_batch, points_number = numpy2blocks_with_seed(
            np.load(numpy_file), 4096)
        xyzi_batch = xyzi_batch.reshape(
            (xyzi_batch.shape[0] * xyzi_batch.shape[1], 4))
        # print('Shape xyzi_batch: ' + str(xyzi_batch.shape))

        # load pred_file
        # pred_file = 'D:\VR3D\code\ML-VR3D\VR3D\labelisation_automatique\Test' \
        #             '\\test\\result\\res_models\chambre_filtre159\chambre_filtre159_pred.npy'
        score_label = np.load(pred_file)
        score_label = score_label.reshape(
            (score_label.shape[0] * score_label.shape[1], 2))
        # print('Shape score_label: ' + str(score_label.shape))
        obj = np.column_stack((xyzi_batch, score_label))
        return obj[0:points_number, :]

    @staticmethod
    def visualize_result(chambre_file, score_label):
        """
        - création fichier .txt (input) post_processing
        - création 2 fichier .las pour la visualisation resultat après detect markings
        :param chambre_file:
        :param score_label:
        :return:chambre_non_marque, chambre_marque
        """
        xyzi = chambre_file.copy()

        xyzi_batch, points_num = numpy2blocks_with_seed(xyzi, 4096)

        obj = np.concatenate((xyzi_batch, score_label), axis=2)
        """for b in range(obj.shape[0]):
            for i in range(obj.shape[1]):
                # color = indoor3d_util.g_label2color[obj[b, i, 5]]
                if obj[b, i, 5] == 0:
                    # obj[b, i, 5] = 0
                    print('NOn chambre detected en [ ' + str(b) + ',' + str(i) + ',5 ]')
                else:
                    obj[b, i, 2] += 1
                    print('Chambre detected')
                    pass
                fout.write('%0.3f %0.3f %0.3f %0.3f %d\n' % (
                obj[b, i, 0], obj[b, i, 1], obj[b, i, 2], obj[b, i, 4] * 1000, obj[b, i, 5]))

        fout.close()"""

        obj_0 = obj[obj[:, :, 5] == 0]
        obj_1 = obj[obj[:, :, 5] != 0]

        obj_1[:, 3] = obj_1[:, 4] * 1000
        obj_0[:, 3] = obj_0[:, 4] * 1000
        """
        fout1 = open(os.path.join(visu_path, filename + '_visu_obj_nM.txt'), 'w')
        fout2 = open(os.path.join(visu_path, filename + '_visu_obj_M.txt'), 'w')
        for b in range(obj.shape[0]):
            for i in range(obj.shape[1]):
                # color = g_label2color[obj[b, i, 5]]
                if obj[b, i, 5] == 0:  # Non marquage

                    fout1.write('%0.3f %0.3f %0.3f %0.3f %d\n' % (obj[b, i, 0], obj[b, i, 1], obj[b, i, 2] + 10,
                                                                  obj[b, i, 4] * 1000,
                                                                  obj[b, i, 5]))
                elif obj[b, i, 5] != 0:  # autre
                    fout2.write('%0.3f %0.3f %0.3f %0.3f %d\n' % (obj[b, i, 0], obj[b, i, 1], obj[b, i, 2],
                                                                  obj[b, i, 4] * 1000,
                                                                  obj[b, i, 5]))
        # Conversion txt2 las
        # print(str(os.path.join(visu_path, filename + '_obj_marquage.txt')))
        # os.system('D:\VR3D\Tools\LAStools\\bin\\txt2las.exe -i \"' + os.path.join(visu_path,
        #                                                                           filename + '_visu_obj_nM.txt\"') + ' -parse xyzi -olas')
        # os.system('D:\VR3D\Tools\LAStools\\bin\\txt2las.exe -i \"' + os.path.join(visu_path,
        #                                                                           filename + '_visu_obj_M.txt\"') + ' -parse xyzi -olas')
        fout1.close()
        fout2.close()
        # shutil.copy(os.path.join(visu_path, filename + '_obj_alls.txt'), '/media/ml/data/Jupyter Notebook/ML_TOPO_MAS')
        # print("copie Done")
        print('Done')"""
        return obj_0[:, :4], obj_1[:, :4]
        return obj_0[:, :4], obj_1[:, :4]
