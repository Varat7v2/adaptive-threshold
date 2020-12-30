import FaceRecognition as FR
import cv2
import glob
import numpy as np
import pickle
import os, sys
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sympy import S, symbols, printing

from facenet import facenet, detect_face
from scipy.optimize import minimize, minimize_scalar

from faceDetection_frozenGraph import TensoflowFaceDector
from myFACENET import FACENET_EMBEDDINGS
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from scipy.interpolate import make_interp_spline, BSpline, CubicSpline
import scipy.integrate as integrate

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import adaptive_config as myconfig

font = {'size': 20}
plt.rc('font', **font)

linearReg = LinearRegression()

sys.setrecursionlimit(10**7) 
print('Recusion Limit: {}'.format(sys.getrecursionlimit()))


# myconfig.ADD_PERSONS = False
# myconfig.SRC_PATH = 'data/data_athelets_test3'
# myconfig.SOURCE = 'image'
# # myconfig.IMAGE_NUM_LIMIT = GALLERY_FVECTORS + 1
# myconfig.IMAGE_NUM_LIMIT = 1
# myconfig.METRIC_BOUND = 0.8

OUTPUT_CSV_PATH = 'results_csv_{}_{}'.format(myconfig.GSIZE, myconfig.DATA_VERSION)
GALLERY_FVECTORS = myconfig.GSIZE

if not os.path.exists(OUTPUT_CSV_PATH):
    os.makedirs(OUTPUT_CSV_PATH)

# myconfig.PATH_TO_CKPT_FACE = 'models/face_ssd_512x512.pb'
# myconfig.PATH_TO_CKPT_FACENET_128D = 'models/facenet-20170511-185253.pb'
# myconfig.PATH_TO_CKPT_FACENET_512D_9905 = 'models/facenet-20180408-102900-CASIA-WebFace.pb'
# myconfig.PATH_TO_CKPT_FACENET_512D_9967 = 'models/faenet-20180402-114759-VGGFace2.pb'

tDetector = TensoflowFaceDector(myconfig.PATH_TO_CKPT_FACE)
myfacenet = FACENET_EMBEDDINGS(myconfig.PATH_TO_CKPT_FACENET_512D_9967)

def serialize_dict(dict_, filename):
    mydict = dict()
    # Load the embeddings of known people or check if there are any new people need to add in the list
    for image in glob.glob('known_persons/*'):
        name = image.split('/')[-1].split('.')[0]
        image = FR.load_image_file(image)
        encoding = FR.face_encoding(image)[0]
        mydict[name] = encoding

    # Serialize dictionary in binary format
    with open(filename, 'wb') as handle:
        pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)

def deserialize_dict(filename):
    # Deserialize dictionary
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def face_distance(face_encoding, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encoding) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encoding - face_to_compare, ord=None, axis=1)


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.9):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

def cosine_similarity(dict_encoding, current_encoding):
    return (np.dot(dict_encoding, current_encoding)/(np.linalg.norm(dict_encoding)*np.linalg.norm(current_encoding)))

def cosine_distance(dict_encoding, current_encoding):
    return (1 - np.dot(dict_encoding, current_encoding)/(np.linalg.norm(dict_encoding)*np.linalg.norm(current_encoding)))

def euclidean_distance(dict_encoding, current_encoding):
    return (np.sqrt(np.sum((dict_encoding - current_encoding) ** 2)))

def facenet_encoding():
    pass

def dlib_encoding():
    pass

def get_facial_embeddings(cropped_faces):
        return myfacenet.run(cropped_faces)

def detect_crop_face(frame, frame_width, frame_height):
    boxes, scores, classes, num_detections = tDetector.run(frame)
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    face_locations = list()
    faces_cropped = list()
    face_boxes = list()

    current_encoding = None

    face_count = 0
    valid_scores = [score for score in scores if score > 0.4]
    num_faces = len(valid_scores)
    # print('No. of faces: {}, score: {}'.format(num_faces, valid_scores))

    for score, box in zip(scores, boxes):
        if score > 0.7 and num_faces == 1:
            # ymin, xmin, ymax, xmax = box
            left = int(box[1]*frame_width)
            top = int(box[0]*frame_height)
            right = int(box[3]*frame_width)
            bottom = int(box[2]*frame_height)

            face_locations.append((top, right, bottom, left))
            face_boxes.append([left, top, right, bottom])
            cropped = frame[top:bottom, left:right]
            # cv2.imwrite('test.jpg', cropped)
            cropped = cv2.resize(cropped, (160,160), interpolation=cv2.INTER_LINEAR)
            faces_cropped.append(facenet.prewhiten(cropped))

            num_curr_faces = len(faces_cropped)
            # eucliDist_matrix = np.zeros((num_curr_faces, num_of_identities))

            if num_curr_faces > 0:
                if myconfig.FACE_RECOGNITION == 'DLIB':
                    face_encoding = FR.face_encoding(frame, face_locations)
                if myconfig.FACE_RECOGNITION == 'FACENET':
                    current_encoding = myfacenet.run(faces_cropped)

                #***********************************************************************************************
                ### DLIB
                # # Loop through each face in this frame of video
                # for (top, right, bottom, left), face_encoding in zip(face_locations, face_encoding):
                #     # See if the face is a match for the known face(s)
                #     matches = FR.compare_faces(known_face_encodings, face_encoding)

                #     name = "Unknown"

                #     # If a match was found in known_face_encodings, just use the first one.
                #     # if True in matches:
                #     #     first_match_index = matches.index(True)
                #     #     name = known_face_names[first_match_index]

                #     # Or instead, use the known face with the smallest distance to the new face
                #     face_distances = FR.face_distance(known_face_encodings, face_encoding)
                #     best_match_index = np.argmin(face_distances)
                #     if matches[best_match_index]:
                #         name = known_face_names[best_match_index]

                # Draw a label with a name below the face
                # cv2.putText(frame, name, (left+5, top-15), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                #************************************************************************************************

            # Draw a box around the face
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),int(round(frame_height/150)), 2)

    return frame, faces_cropped, face_boxes, current_encoding

def find_threshold_stats(mydirs):
    mydict_list = list()
    # print(mydirs)
    for dir_name in mydirs:
        if len(list(f for f in glob.glob(os.path.join(myconfig.SRC_PATH, dir_name)+'/*'))) >= myconfig.IMAGE_NUM_LIMIT:
            cropped_faces = list()
            faces_embeddings = list()
            mydict = dict()
            count = 0
            for file in glob.glob(os.path.join(myconfig.SRC_PATH, dir_name)+'/*'):
                filename = file.split('/')[-1]
                frame = cv2.imread(file)
                frame_height, frame_width, _ = frame.shape
                #check if there are two faces in the frame
                frame, cropped, face_boxes, current_encoding = detect_crop_face(frame, frame_width, frame_height)
                if cropped is not None and current_encoding is not None:
                    cropped_faces.append(cropped)
                    faces_embeddings.append(current_encoding)

                count += 1

            # print()
            if len(faces_embeddings) > 0:
                # faces_embeddings.append(get_facial_embeddings(cropped_faces))
                mydict['face_id'] = dir_name
                mydict['face_embeddings'] = faces_embeddings
                mydict_list.append(mydict)

    # Check if dictionary is empty
    if len(mydict_list) == 0:
        raise Exception("The size of the dictionary: mydict is 0")
    else:
        print('dictionary size: {}'.format(len(mydict_list)))

    database_face_ids = list()
    database_face_embeddings = list()
    for dict_ in mydict_list:
        # print(dict_['face_embeddings'].shape)
        # # sys.exit(0)
        database_face_ids.append(dict_['face_id'])
        # database_face_embeddings.append(np.squeeze(dict_['face_embeddings'], axis=0))
        database_face_embeddings.append(dict_['face_embeddings'])

    # stats_menu = ['min', 'max', 'mean', 'std', 'variance']
    threshold_auto_dict = dict()
    threshold_cross_dict = dict()
    threshold_dict = dict()
    threshold_auto_min = list()
    threshold_cross_min = list()
    threshold_auto_avg = list()
    threshold_cross_avg = list()

    auto_distance = list()
    cross_distance = list()
    auto_similarity = list()
    cross_similarity = list()

    auto_min, cross_min = list(), list()
    auto_max, cross_max = list(), list()
    auto_avg, cross_avg = list(), list()
    auto_var, cross_var = list(), list()
    auto_std, cross_std = list(), list()

    P_same = 0
    P_diff = 0
    for i, name, dict_encodings in tqdm(zip(np.arange(len(database_face_ids)), 
                                                        database_face_ids, 
                                                        database_face_embeddings)):
        for j, dict_encodings_copy in zip(np.arange(len(database_face_ids)), database_face_embeddings):
            for k, (dict_encoding) in enumerate(zip(dict_encodings)):
                dict_encoding = np.squeeze(np.asarray(dict_encoding), axis=0)
                for l, (dict_encoding_copy) in enumerate(zip(dict_encodings_copy)):
                    dict_encoding_copy = np.squeeze(np.asarray(dict_encoding_copy), axis=0).transpose()
                    if i == j:
                        if k != l:
                            P_same += 1
                            auto_similarity.append(cosine_similarity(dict_encoding, dict_encoding_copy).tolist()[0][0])
                    else:
                        P_diff += 1
                        cross_similarity.append(cosine_similarity(dict_encoding, dict_encoding_copy).tolist()[0][0])

    # auto_similarity = np.array(auto_similarity)
    # cross_similarity = np.array(cross_similarity)
    # print('Auto-similarity size: {}, cross-similarity size: {}'.format(auto_similarity.shape, cross_similarity.shape))
    try:
        auto_min.append(np.min(auto_similarity))
        auto_max.append(np.max(auto_similarity))
        auto_avg.append(np.mean(auto_similarity))
        auto_var.append(np.var(auto_similarity))
        auto_std.append(np.std(auto_similarity))
        # auto_stats = [auto_min, auto_max, auto_avg, auto_var, auto_std]
        # threshold_auto_min.append(auto_min)
        # threshold_auto_avg.append(auto_avg)

        cross_min.append(np.min(cross_similarity))
        cross_max.append(np.max(cross_similarity))
        cross_avg.append(np.mean(cross_similarity))
        cross_var.append(np.var(cross_similarity))
        cross_std.append(np.std(cross_similarity))
        # cross_stats = [cross_min, cross_max, cross_avg, cross_var, cross_std]
        # threshold_cross_min.append(cross_min)
        # threshold_cross_avg.append(cross_avg)

        # threshold_dict[name] = np.asarray([auto_stats, cross_stats], dtype=np.float32)

    except ValueError: #raised if auto_dist and cross_dist are empty.
        pass

    # # calculate parameters
    # sample_mean = mean(cross_similarity)
    # sample_std = std(cross_similarity)
    # print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))

    # # define the distribution
    # dist = norm(sample_mean, sample_std)

    auto_mean = np.mean(auto_similarity)
    auto_std = np.std(auto_similarity)
    auto_var = np.var(auto_similarity)
    auto_min = np.min(auto_similarity)
    auto_max = np.max(auto_similarity)

    cross_mean = np.mean(cross_similarity)
    cross_std = np.std(cross_similarity)
    cross_var = np.var(cross_similarity)
    cross_min = np.min(cross_similarity)
    cross_max = np.max(cross_similarity)

    A = auto_var - cross_var
    B = 2*(auto_mean*cross_var - cross_mean*auto_var)
    C = np.square(cross_mean)*auto_var - np.square(auto_mean)*cross_var - 2*auto_var*cross_var*np.log(auto_std/cross_std)
    coeff = [A, B, C]
    roots = [x for x in np.roots(coeff) if x>-0 and x<=1]
    # print(roots)

    if PLOT_FLAG:
        # Density Plot and Histogram of all arrival delays
        fig1 = plt.figure('histogram-gaussian-curve')
        ax1 = sns.distplot(auto_similarity, norm_hist=True, kde=True, 
                     bins=int(180/5), color = 'blue', 
                     hist_kws={'edgecolor':'black'},
                     label='auto_similarity',
                     kde_kws={'linewidth': 2, 'linestyle':'--', 'color':'blue', 'alpha':0.8})
        ax1.set(xlabel='Threshold', ylabel='probability density')

        # Density Plot and Histogram of all arrival delays
        ax1 = sns.distplot(cross_similarity, norm_hist=True, kde=True, 
                     bins=int(180/5), color = 'red',
                     hist_kws={'edgecolor':'black'},
                     label='cross_similarity',
                     kde_kws={'linewidth': 2, 'linestyle':'--', 'color':'red', 'alpha':0.8})
        ax1.set(xlabel='Threshold', ylabel='density function')

        auto_x = np.linspace(auto_min-0.1, auto_max+0.2, 1000)
        auto_y = scipy.stats.norm.pdf(auto_x, auto_mean, auto_std)
        # auto_ymin = np.min(auto_y)
        # auto_ymax = np.max(auto_y)
        # auto_y = (auto_y-auto_ymin)/(auto_ymax-auto_ymin)
        ax1.plot(auto_x, auto_y, color='blue', linewidth=3, label='auto_gaussian')

        cross_x = np.linspace(cross_min-0.1, cross_max+0.2, 1000)
        cross_y = scipy.stats.norm.pdf(cross_x, cross_mean, cross_std)
        # cross_ymin = np.min(cross_y)
        # cross_ymax = np.max(cross_y)
        # cross_y = (cross_y-cross_ymin)/(cross_ymax-cross_ymin)
        ax1.plot(cross_x, cross_y, color='red', linewidth=3, label='cross_gaussian')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=4)
        # h1, l1 = ax1.get_legend_handles_labels()
        # h2, l2 = ax2.get_legend_handles_labels()
        # ax.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=6)

        # # f = [np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance)) for x in auto_similarity]
        # # plt.plot(np.arange(len(auto_similarity)), f)

        # # auto_var = np.var(auto_similarity)
        # # cross_var = np.var(cross_similarity)
        # # k1 = auto_var/(auto_var+cross_var)
        # # k2 = cross_var/(auto_var+cross_var)
        # # updated_similarity = k2*np.array(auto_similarity) + k1*np.array(cross_similarity)

        # # Density Plot and Histogram of all arrival delays
        # sns.distplot(updated_similarity, hist=True, kde=True, 
        #              bins=int(180/5), color = 'green', 
        #              hist_kws={'edgecolor':'black'},
        #              kde_kws={'linewidth': 4})

        fig2 = plt.figure('auto-similarity distribution')
        plt.scatter(np.arange(len(auto_similarity)), auto_similarity, s=10, label='auto_similarity', color='deepskyblue')
        plt.plot(np.arange(len(auto_similarity)), [auto_avg]*len(auto_similarity), label='auto_average', linewidth=3, color='green')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=2)
        plt.xlabel('No. of similarity pairing')
        plt.ylabel('Cosine similarity')
        # plt.legend(loc="best")
        # auto_dist_1 = np.asarray(auto_dist_1)

        fig3 = plt.figure('cross-similarity distribution')
        plt.scatter(np.arange(len(cross_similarity)), cross_similarity, s=10, label='cross_similarity', color='coral')
        plt.plot(np.arange(len(cross_similarity)), [cross_avg]*len(cross_similarity), label='cross_average', linewidth=3, color='green')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=2)
        plt.xlabel('No. of similarity pairing')
        plt.ylabel('Cosine similarity')
        # plt.legend(loc="best")
        # cross_dist_1 = np.asarray(cross_dist_1)

        # Creating histogram 
        # fig, axs = plt.subplots(1, 1, fimyconfig.GSIZE =(10, 7), tight_layout = True) 
        
        # axs.hist(auto_similarity, bins = 10)
        # axs.hist(auto_similarity, bins = 10)
        # fig32 = plt.figure('Histogram')
        bins = np.linspace(-1, 2, 100)
        # plt.figure('Histogram')
        fig10, ax10_1 = plt.subplots(num='Histogram')
        ax10_2 = ax10_1.twinx()
        ax10_2.hist(auto_similarity, bins, alpha=0.7, label='auto_similarity', color='deepskyblue')
        ax10_2.set_ylabel('Auto-similarity distribution', fontsize=20)
        ax10_2.set_xlabel('Cosine similarity', fontsize=20)
        # plt.figure('Cross similarity histogram')
        ax10_1.hist(cross_similarity, bins, alpha=0.8, label='cross_similarity', color='coral')
        ax10_1.set_ylabel('Cross-similarity distribution', fontsize=20)
        ax10_1.set_xlabel('Cosine similarity', fontsize=20)
        # plt.xlabel('Cosine similarity')
        # plt.ylabel('No. of occurances')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=2)
        h10_1, l10_1 = ax10_1.get_legend_handles_labels()
        h10_2, l10_2 = ax10_2.get_legend_handles_labels()
        ax10_2.legend(h10_1+h10_2, l10_1+l10_2, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=6)

        # cross_dist_1 = np.asarray(cross_dist_1)

        # plt.figure('auto cosine similarity')
        # plt.plot(np.arange(len(auto_similarity)), [auto_avg]*len(auto_similarity), label='auto_average')

        # # FINDING THE THRESHOLD
        # myauto_min = np.min(auto_similarity)
        # mycross_max = np.max(cross_similarity)
        # myauto_avg = np.mean(auto_similarity)
        # mycross_avg = np.mean(cross_similarity)

        # if myauto_min >= mycross_max:
        #     adaptive_threshold = (myauto_min + mycross_max) / 2
        # else:
        #     adaptive_threshold = myauto_min

        # print('Threshold={}'.format(adaptive_threshold))
        # print('Auto Similarity # = {}, Cross_similarity # = {}'.format(len(auto_similarity), len(cross_similarity)))
        # plt.show()


    # if np.max(threshold_auto_min) < np.min(threshold_cross_min): # safest selection
    #     threshold_face = max(threshold_auto_min)
    #     status = 'Best'
    # else:
    #     if np.mean(threshold_auto_avg) < np.min(threshold_cross_avg):
    #         threshold_face = np.mean(threshold_auto_avg)
    #         status = 'Good'
    #     elif np.mean(threshold_auto_avg) < np.mean(threshold_cross_avg):
    #         threshold_face = np.mean(threshold_auto_avg)
    #         status = 'Fine'
    #     else:
    #         threshold_face = np.min(threshold_cross_avg)    #risky selection among above criterions
    #         status = 'Risky'
    
    # print('\n')
    # print('Min of auto_avg: {}'.format(np.min(threshold_auto_avg)))
    # print('Max of auto_avg: {}'.format(np.max(threshold_auto_avg))) # NO USE
    # print('Min of cross_avg: {}'.format(np.min(threshold_cross_avg)))
    # print('Max of cross_avg: {}'.format(np.max(threshold_cross_avg)))
    # print('Avg of auto_avg: {}'.format(np.mean(threshold_auto_avg)))
    # print('Avg of cross_avg: {}'.format(np.mean(threshold_cross_avg)))
    # print('\n')
    # print('Min of auto_min: {}'.format(np.min(threshold_auto_min))) # NO USE
    # print('Max of auto_min: {}'.format(np.max(threshold_auto_min)))
    # print('Avg of auto_min: {}'.format(np.mean(threshold_auto_min)))
    # print('Min of cross_min: {}'.format(np.min(threshold_cross_min)))
    # print('Max of cross_min: {}'.format(np.max(threshold_cross_min))) # NO USE
    # print('Avg of cross_min: {}'.format(np.mean(threshold_cross_min)))
    # print('\n')
    # print('Threshold_face: {} ({})'.format(threshold_face, status))
    # print('\n')

    # file = open("face_threshold.txt","w") 
    # file.write('Min of auto_avg: {} \n'.format(np.min(threshold_auto_avg)))
    # file.write('Max of auto_avg: {} \n'.format(np.max(threshold_auto_avg))) # NO USE
    # file.write('Min of cross_avg: {} \n'.format(np.min(threshold_cross_avg)))
    # file.write('Max of cross_avg: {} \n'.format(np.max(threshold_cross_avg)))
    # file.write('Avg of auto_avg: {} \n'.format(np.mean(threshold_auto_avg)))
    # file.write('Avg of cross_avg: {} \n'.format(np.mean(threshold_cross_avg)))
    # file.write('\n')
    # file.write('Min of auto_min: {} \n'.format(np.min(threshold_auto_min))) # NO USE
    # file.write('Max of auto_min: {} \n'.format(np.max(threshold_auto_min)))
    # file.write('Avg of auto_min: {} \n'.format(np.mean(threshold_auto_min)))
    # file.write('Min of cross_min: {} \n'.format(np.min(threshold_cross_min)))
    # file.write('Max of cross_min: {} \n'.format(np.max(threshold_cross_min))) # NO USE
    # file.write('Avg of cross_min: {} \n'.format(np.mean(threshold_cross_min)))
    # file.write('\n')
    # file.write('Threshold_face: {} ({})'.format(threshold_face, status))
    # file.close()

    # # # name_list, threshold_stats = list(), list()
    # # # 0->min, 1->max, 2->avg, 3->variance, 4->standard deviation
    # # for name, details in threshold_dict.items():
    # #     # print(name)
    # #     threshold_stats = np.squeeze(details, axis=2).tolist()
    # #     auto_stats = threshold_stats[0]
    # #     cross_stats = threshold_stats[1]
    # #     # print(auto_stats)
    # #     # print(cross_stats)

    return max(roots), auto_mean, cross_mean, P_same, P_diff


# def compare_embeddings():
#     for i, current_encoding, facebox in zip(np.arange(len(current_encoding)).tolist(), current_encoding, face_boxes):
#         dist_updated = list()
#         for j, dict_name, dict_encodings in zip(np.arange(num_of_identities), known_face_names, known_face_encodings):
#             dist = list()
#             for dict_encoding in dict_encodings:
#                 # dist.append(euclidean_distance(dict_encoding, current_encoding))
#                 dist.append(cosine_similarity(dict_encoding, current_encoding))
#             # eucliDist_matrix[i][j] = np.min(np.asarray(dist, dtype=np.float32))
            
#             dist_updated.append(max(dist))

#         if max(dist_updated) > 0.7:
#             name = known_face_names[np.argmax(dist_updated)]
#         else:
#             name = 'Unknown'

def evaluate_model(query_dict, gallery_dict, adaptive_threshold, P_same, P_diff):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    mygallery_list = list()
    for q_key, q_values in tqdm(query_dict.items()):
        # dist_updated = list()
        for q_embedding in q_values:
            known_names = list()
            dist = list()
            mygallery = dict()
            for g_key, g_values in gallery_dict.items():
                for g_embedding in g_values:
                    known_names.append(g_key)
                    # dist.append(euclidean_distance(dict_encoding, current_encoding))
                    dist.append(cosine_similarity(q_embedding, g_embedding))

                    # mygallery['q_key'] = q_key
                    # mygallery['known_names'] = known_names
                    # mygallery['dist'] = dist
                    # mygallery_list.append(mygallery)
            
            # dist_updated.append(max(dist))
            max_dist = max(dist)
            name = known_names[np.argmax(dist)]
            if max_dist >= adaptive_threshold:
                if name == q_key:
                    tp += 1
                else:
                    fp += 1
            else:
                if name == q_key:
                    fn += 1
                else:
                    tn += 1

    precision = tp/(tp+fp+1e-7)
    recall = tp/(tp+fn+1e-7)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    f1score = 2*(precision*recall)/(precision+recall+1e-7)

    # adaptive_precisions.append(precision)
    # adaptive_recalls.append(recall)
    # adaptive_f1scores.append(f1score)

    # print('TP={}, FP={}, TN={}, FN={}'.format(tp, fp, tn, fn))

    # VAL = tp/P_same
    # FAR = fp/P_diff

    tpr = tp/(tp+fn+1e-7)
    fpr = fp/(fp+tn+1e-7)
    
    return precision, recall, f1score, accuracy, tpr, fpr


def obj_func(x):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    # for mydict_ in mygallery_list:
    #     known_names = mydict_['known_names']
    #     dist = mydict_['dist']
    #     q_key = mydict_['q_key']
    
    for q_key, q_values in tqdm(query_dict.items()):
        # dist_updated = list()
        for q_embedding in q_values:
            known_names = list()
            dist = list()
            mygallery = dict()
            for g_key, g_values in gallery_dict.items():
                for g_embedding in g_values:
                    known_names.append(g_key)
                    # dist.append(euclidean_distance(dict_encoding, current_encoding))
                    dist.append(cosine_similarity(q_embedding, g_embedding))
            
            # dist_updated.append(max(dist))
            max_dist = max(dist)
            name = known_names[np.argmax(dist)]
            if max_dist >= x:
                if name == q_key:
                    tp += 1
                else:
                    fp += 1
            else:
                if name == q_key:
                    fn += 1
                else:
                    tn += 1

        precision = tp/(tp+fp+1e-7)
        recall = tp/(tp+fn+1e-7)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        f1score = 2*(precision*recall)/(precision+recall+1e-7)

        # adaptive_precisions.append(precision)
        # adaptive_recalls.append(recall)
        # adaptive_f1scores.append(f1score)

        tpr = abs(tp)/abs(tp+fn+1e-7) #VAL
        fpr = abs(fp)/abs(fp+tn+1e-7) #FAR
        # print(abs(tpr-fpr))
    return (tpr-fpr)
    # return f1score

def myquadeqn(mypoly, x):
    return mypoly[0]*x**2+mypoly[1]*x+mypoly[2]

def main():
    global query_dict, gallery_dict, PLOT_FLAG
    query_dict = dict()
    gallery_dict = dict()
    print('Creating gallery and query dictionary...')
    auto_means = list()
    cross_means = list()

    updated_dirs = list()
    directories = [mydir for mydir in os.listdir(myconfig.SRC_PATH)]
    for d in directories:
        if len(list(f for f in glob.glob(os.path.join(myconfig.SRC_PATH, d)+'/*'))) >= myconfig.IMAGE_NUM_LIMIT:
            updated_dirs.append(d)

    adaptive_precisions, adaptive_recalls, adaptive_f1scores, adaptive_thresholds = list(), list(), list(), list()
    fixed_thresholds_1, fixed_precisions_1, fixed_recalls_1, fixed_f1scores_1 = list(), list(), list(), list()
    fixed_thresholds_2, fixed_precisions_2, fixed_recalls_2, fixed_f1scores_2 = list(), list(), list(), list()
    fixed_thresholds_3, fixed_precisions_3, fixed_recalls_3, fixed_f1scores_3 = list(), list(), list(), list()

    TPRs_adaptive, FPRs_adaptive = [0,0,0,0,0,0,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1]
    TPRs_fixed_1, FPRs_fixed_1   = [0,0,0,0,0,0,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1]
    TPRs_fixed_2, FPRs_fixed_2   = [0,0,0,0,0,0,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1]
    TPRs_fixed_3, FPRs_fixed_3   = [0,0,0,0,0,0,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1]

    # TPRs_adaptive, FPRs_adaptive = list(), list()
    # TPRs_fixed_1, FPRs_fixed_1   = list(), list()
    # TPRs_fixed_2, FPRs_fixed_2   = list(), list()
    # TPRs_fixed_3, FPRs_fixed_3   = list(), list()

    loop_count = 0
    dir_count = 0
    PLOT_FLAG = False

    # precision = 0
    # recall = 0
    # f1score = 0
    ptotal, pcount, pfixed_count_1, pfixed_count_2, pfixed_count_3 = 0, 0, 0, 0, 0
    mydirs = list()
    num_people = 0

    # OPTIMIZATION_METRIC = f1score
    fixed_threshold_min = 0.3
    fixed_threshold_default = 0.5
    fixed_threshold_max = 0.7

    print('UPDARED DIRS SIZE: {}'.format(len(updated_dirs)))

    for d in updated_dirs:
        loop_count += 1
        if loop_count == len(updated_dirs):
            PLOT_FLAG = True
        num_people += 1
        if dir_count < 1:
            mydirs.append(d)
            dir_count += 1
            # continue
        else:
            mydirs.append(d)
            for mydir in mydirs:
                query_embeddings = list()
                for img in glob.glob(os.path.join(myconfig.SRC_PATH, mydir)+'/*'):
                    frame = cv2.imread(img)
                    try:
                        frame_height, frame_width, _ = frame.shape
                        frame, faces_cropped, face_boxes, current_encoding = detect_crop_face(frame, 
                                                                                              frame_width, 
                                                                                              frame_height)

                        if current_encoding is not None:
                            query_embeddings.append(current_encoding[0])
                    except AttributeError:
                        print("Image is None. Couldn't find shape!")
                        continue
                        #code to move to next frame

                if len(query_embeddings) != 0:
                    # WE CAN INCREASE NUMBER OF IMAGES IN GALLERY TO INCREASE PRECISION AND ACCURACY
                    indexes = np.arange(len(query_embeddings)).tolist()
                    rand_indexes = random.sample(indexes, k=GALLERY_FVECTORS)
                    for item in rand_indexes:
                        indexes.remove(item)

                    query_dict[mydir] = [query_embeddings[i] for i in indexes]
                    gallery_dict[mydir] = [query_embeddings[i] for i in rand_indexes]

            
            # FINDING OPTIMAL THRESHOLD VALUE
            adaptive_threshold, auto_mean, cross_mean, P_same, P_diff = find_threshold_stats(mydirs)
            # print(adaptive_threshold)
            # adaptive_thresholds.append(adaptive_threshold)
            auto_means.append(auto_mean)
            cross_means.append(cross_mean)
            # print('Adaptive threshold: {}'.format(adaptive_threshold))

            # # SERIALIZING GALLERY AND QUERY EMBEDDINGS
            # serialize_dict(gallery_dict, 'data/gallery_dict.pkl')
            # serialize_dict(query_dict, 'data/query_dict.pkl')

            # ADAPTIVE THRESHOLD
            adaptive_precision, adaptive_recall, adaptive_f1score, adaptive_accuracy, adaptive_tpr, adaptive_fpr = evaluate_model(
                query_dict, gallery_dict, adaptive_threshold, P_same, P_diff)

            adaptive_threshold_prev = adaptive_threshold
            adaptive_precision_prev = adaptive_precision
            adaptive_recall_prev = adaptive_recall
            adaptive_f1score_prev = adaptive_f1score
            adaptive_tpr_prev = adaptive_tpr
            adaptive_fpr_prev = adaptive_fpr
            adaptive_accuracy_prev = adaptive_accuracy
            ptotal += 1
            if adaptive_f1score < myconfig.METRIC_BOUND:
            # if False:
                print('optimization going on ...')
                myfunc = lambda x: -obj_func(x)
                # res = minimize_scalar(myfunc, bounds=(cross_mean, auto_mean), method='bounded')
                res = minimize_scalar(myfunc, bounds=(0, 1), method='bounded')
                threshold_updated = res.x
                adaptive_precision, adaptive_recall, adaptive_f1score, adaptive_accuracy, adaptive_tpr, adaptive_fpr = evaluate_model(
                    query_dict, gallery_dict, threshold_updated, P_same, P_diff)
                if adaptive_f1score >= myconfig.METRIC_BOUND:
                    pcount += 1
                if adaptive_f1score >= adaptive_f1score_prev:
                    print('Seletected optimized threshold!')
                    opt_threshold = threshold_updated
                    # print('cross mean={}, auto mean={}, opt_threshold={}'.format(cross_mean, auto_mean, opt_threshold))
                else:
                    opt_threshold = adaptive_threshold_prev
                    adaptive_precision = adaptive_precision_prev
                    adaptive_recall = adaptive_recall_prev
                    adaptive_f1score = adaptive_f1score_prev
                    adaptive_tpr = adaptive_tpr_prev
                    adaptive_fpr = adaptive_fpr_prev
                    adaptive_accuracy = adaptive_accuracy_prev
            else:
                print('taking normal threshold ...')
                pcount += 1
                opt_threshold = adaptive_threshold

            # Evaluating model after searching optimal threshold
            # precision, recall, f1score, accuracy, tpr, fpr = evaluate_model(query_dict, gallery_dict, opt_threshold, P_same, P_diff)
            adaptive_precisions.append(adaptive_precision)
            adaptive_recalls.append(adaptive_recall)
            adaptive_f1scores.append(adaptive_f1score)
            adaptive_thresholds.append(opt_threshold)
            if adaptive_fpr>0 and adaptive_fpr<1:
                TPRs_adaptive.append(adaptive_tpr)
                FPRs_adaptive.append(adaptive_fpr)
            elif adaptive_fpr == 0:
                TPRs_adaptive.append(0)
                FPRs_adaptive.append(0)
            elif adaptive_fpr == 1:
                TPRs_adaptive.append(1)
                FPRs_adaptive.append(1)

            # FIXED THRESHOLD
            # fixed threshold --> lower bound (1)
            precision_fixed_1, recall_fixed_1, f1score_fixed_1,accuracy_fixed1, tpr_fixed_1, fpr_fixed_1 = evaluate_model(
                query_dict, gallery_dict, fixed_threshold_min, P_same, P_diff)
            fixed_precisions_1.append(precision_fixed_1)
            fixed_recalls_1.append(recall_fixed_1)
            fixed_f1scores_1.append(f1score_fixed_1)
            fixed_thresholds_1.append(fixed_threshold_min)
            if fpr_fixed_1>0 and fpr_fixed_1<1:
                TPRs_fixed_1.append(tpr_fixed_1)
                FPRs_fixed_1.append(fpr_fixed_1)
            elif fpr_fixed_1 == 0:
                TPRs_fixed_1.append(0)
                FPRs_fixed_1.append(0)
            elif fpr_fixed_1 == 1:
                TPRs_fixed_1.append(1)
                FPRs_fixed_1.append(1)
            if  f1score_fixed_1 >= myconfig.METRIC_BOUND:
                pfixed_count_1 += 1
            
            # fixed threshold --> middle bound (2)
            precision_fixed_2, recall_fixed_2, f1score_fixed_2, accuracy_fixed2, tpr_fixed_2, fpr_fixed_2 = evaluate_model(
                query_dict, gallery_dict, fixed_threshold_default, P_same, P_diff)
            fixed_precisions_2.append(precision_fixed_2)
            fixed_recalls_2.append(recall_fixed_2)
            fixed_f1scores_2.append(f1score_fixed_2)
            if fpr_fixed_2>0 and fpr_fixed_2<1:
                TPRs_fixed_2.append(tpr_fixed_2)
                FPRs_fixed_2.append(fpr_fixed_2)
            if fpr_fixed_2 == 0:
                TPRs_fixed_2.append(0)
                FPRs_fixed_2.append(0)
            if fpr_fixed_2 == 1:
                TPRs_fixed_2.append(1)
                FPRs_fixed_2.append(1)
            fixed_thresholds_2.append(fixed_threshold_default)
            if  f1score_fixed_2 >= myconfig.METRIC_BOUND:
                pfixed_count_2 += 1

            # fixed threshold --> upper bound (3)
            precision_fixed_3, recall_fixed_3, f1score_fixed_3,accuracy_fixed3, tpr_fixed_3, fpr_fixed_3 = evaluate_model(
                query_dict, gallery_dict, fixed_threshold_max, P_same, P_diff)
            fixed_precisions_3.append(precision_fixed_3)
            fixed_recalls_3.append(recall_fixed_3)
            fixed_f1scores_3.append(f1score_fixed_3)
            if fpr_fixed_3>0 and fpr_fixed_3<1:
                TPRs_fixed_3.append(tpr_fixed_3)
                FPRs_fixed_3.append(fpr_fixed_3)
            if fpr_fixed_3 == 0:
                TPRs_fixed_3.append(0)
                FPRs_fixed_3.append(0)
            if fpr_fixed_3 == 1:
                TPRs_fixed_3.append(1)
                FPRs_fixed_3.append(1)
            fixed_thresholds_3.append(fixed_threshold_max)
            if  f1score_fixed_3 >= myconfig.METRIC_BOUND:
                pfixed_count_3 += 1

            # print(len(VALs))
            # print(len(FARs))
            # print('Final Optimum Threshold={}'.format(opt_threshold))
            # print('VAL={}, FAR={}'.format(VAL*100, FAR*100))
            # print('Precision={}, Recall={}, F1score={}'.format(precision*100, recall*100, f1score*100))

    # print('PRECISION SIZE: {}'.format(len(adaptive_precisions)))
    # print('Adaptive threshold size: {}'.format(len(adaptive_thresholds)))
    df_adaptive = pd.DataFrame(list(zip(adaptive_thresholds, adaptive_precisions, adaptive_recalls, adaptive_f1scores)),
        columns=['Adaptive-Threshold', 'Precision', 'Recall', 'F1-score'])
    df_fixed_1 = pd.DataFrame(list(zip(fixed_thresholds_1, fixed_precisions_1, fixed_recalls_1, fixed_f1scores_1)),
        columns=['Fixed-Threshold=0.3', 'Precision@threshold=0.3', 'Recall@threshold=0.3', 'F1-score@threshold=0.3'])
    df_fixed_2 = pd.DataFrame(list(zip(fixed_thresholds_2, fixed_precisions_2, fixed_recalls_2, fixed_f1scores_2)),
        columns=['Fixed-Threshold=0.5', 'Precision@threshold=0.5', 'Recall@threshold=0.5', 'F1-score@threshold=0.5'])
    df_fixed_3 = pd.DataFrame(list(zip(fixed_thresholds_3, fixed_precisions_3, fixed_recalls_3, fixed_f1scores_3)),
        columns=['Fixed-Threshold=0.7', 'Precision@threshold=0.7', 'Recall@threshold=0.7', 'F1-score@threshold=0.7'])
    
    df_roc_adaptive = pd.DataFrame(list(zip(FPRs_adaptive, TPRs_adaptive)), columns=['FPR', 'TPR'])
    df_roc_fixed_1 = pd.DataFrame(list(zip(FPRs_fixed_1, TPRs_fixed_1)), columns=['FPR@threshold=0.3', 'TPR@threshold=0.3'])
    df_roc_fixed_2 = pd.DataFrame(list(zip(FPRs_fixed_2, TPRs_fixed_3)), columns=['FPR@threshold=0.5', 'TPR@threshold=0.5'])
    df_roc_fixed_3 = pd.DataFrame(list(zip(FPRs_fixed_3, TPRs_fixed_3)), columns=['FPR@threshold=0.7', 'TPR@threshold=0.7'])
    
    df_adaptive.to_csv('data/results_csv/adaptive_threshold.csv')
    df_fixed_1.to_csv('data/results_csv/fixed_threshold_1.csv')
    df_fixed_2.to_csv('data/results_csv/fixed_threshold_2.csv')
    df_fixed_3.to_csv('data/results_csv/fixed_threshold_3.csv')

    df_roc_adaptive.to_csv('data/results_csv/roc_curve_adaptive.csv')
    df_roc_fixed_1.to_csv('data/results_csv/roc_curve@threshold=0.3.csv')
    df_roc_fixed_2.to_csv('data/results_csv/roc_curve@threshold=0.5.csv')
    df_roc_fixed_3.to_csv('data/results_csv/roc_curve@threshold=0.7.csv')

    print('Percentage of precision@adaptive_threshold >= {}: {}%'.format(myconfig.METRIC_BOUND, pcount/(ptotal+1e-7)*100))
    print('Percentage of precision@fixed_threshold=0.3 >= {}: {}%'.format(myconfig.METRIC_BOUND, pfixed_count_1/(ptotal+1e-7)*100))
    print('Percentage of precision@fixed_threshold=0.5 >= {}: {}%'.format(myconfig.METRIC_BOUND, pfixed_count_2/(ptotal+1e-7)*100))
    print('Percentage of precision@fixed_threshold=0.7 >= {}: {}%'.format(myconfig.METRIC_BOUND, pfixed_count_3/(ptotal+1e-7)*100))
    print('**************MODEL ACCURACY*********************')
    print('Percentage of accuracy@adaptive_threshold: {}%'.format(adaptive_accuracy*100))
    print('Percentage of accuracy@fixed_threshold=0.3: {}%'.format(accuracy_fixed1*100))
    print('Percentage of accuracy@fixed_threshold=0.5: {}%'.format(accuracy_fixed2*100))
    print('Percentage of accuracy@fixed_threshold=0.7: {}%'.format(accuracy_fixed3*100))
    print('Number of pople taken into account: {}'.format(num_people))
    people_count = np.arange(len(adaptive_thresholds))+2
    # print('People count: {}'.format(people_count))

    fig4, ax4 = plt.subplots(num='roc_curve')
    ax4.scatter(FPRs_adaptive, TPRs_adaptive, s=100, color='blue')
    ax4.scatter(FPRs_fixed_1, TPRs_fixed_1, s=100, marker='s', color='darkorange')
    ax4.scatter(FPRs_fixed_2, TPRs_fixed_2, s=100, marker='*', color='green')
    ax4.scatter(FPRs_fixed_3, TPRs_fixed_3, s=100, marker='^', color='red')
    ax4.set_ylim([0.0,1.0])
    ax4.set_xlim([0,1])
    # print(len('TPRS_adaptive size={}, FPRs_adaptive size={}'.format(TPRs_adaptive, FPRs_adaptive)))
    mypoly = np.polyfit(FPRs_adaptive, TPRs_adaptive, 2)
    mymodel = np.poly1d(mypoly)
    new_FPRs = np.linspace(0, 1, 200)
    TPRs_predicted = mymodel(new_FPRs)
    # Estimating ROC-area under the curve
    x = new_FPRs
    auc_adaptive = integrate.quad(lambda x: myquadeqn(mypoly, x), 0.0, 1.0)[0]
    # print('AUC: {}'.format(auc))

    ax4.plot(new_FPRs, TPRs_predicted, label='adaptive_threshold', linewidth=3, color='blue')

    # mymodel1 = np.poly1d(np.polyfit(FPRs_fixed_1, TPRs_fixed_1, 2))
    # TPRs_predicted_fixed_1 = mymodel1(new_FPRs)
    # ax4.plot(new_FPRs, TPRs_predicted_fixed_1, label='fixed_threshold=0.3', linewidth=3, color='darkorange')

    # mymodel2 = np.poly1d(np.polyfit(FPRs_fixed_2, TPRs_fixed_2, 2))
    # TPRs_predicted_fixed_2 = mymodel2(new_FPRs)
    # ax4.plot(new_FPRs, TPRs_predicted_fixed_2, label='fixed_threshold=0.5', linewidth=3, color='green')

    # mymodel3 = np.poly1d(np.polyfit(FPRs_fixed_3, TPRs_fixed_3, 2))
    # TPRs_predicted_fixed_3 = mymodel3(new_FPRs)
    # ax4.plot(new_FPRs, TPRs_predicted_fixed_3, label='fixed_threshold=0.7', linewidth=3, color='red')

    mypoly1 = np.polyfit(FPRs_fixed_1, TPRs_fixed_1, 2)
    mymodel1 = np.poly1d(mypoly1)
    TPRs_predicted_fixed_1 = mymodel1(new_FPRs)
    auc_fixed1 = integrate.quad(lambda x: myquadeqn(mypoly1, x), 0.0, 1.0)[0]
    ax4.plot(new_FPRs, TPRs_predicted_fixed_1, label='fixed_threshold=0.3', linewidth=3, color='darkorange', linestyle='dashed')

    mypoly2 = np.polyfit(FPRs_fixed_2, TPRs_fixed_2, 2)
    mymodel2 = np.poly1d(mypoly2)
    TPRs_predicted_fixed_2 = mymodel2(new_FPRs)
    auc_fixed2 = integrate.quad(lambda x: myquadeqn(mypoly2, x), 0.0, 1.0)[0]
    ax4.plot(new_FPRs, TPRs_predicted_fixed_2, label='fixed_threshold=0.5', linewidth=3, color='green', linestyle='dashed')

    # print(FPRs_fixed_3)
    # print(TPRs_fixed_3)
    # sys.exit(0)
    
    mypoly3 = np.polyfit(FPRs_fixed_3, TPRs_fixed_3, 2)
    mymodel3 = np.poly1d(mypoly3)
    TPRs_predicted_fixed_3 = mymodel3(new_FPRs)
    auc_fixed3 = integrate.quad(lambda x: myquadeqn(mypoly3, x), 0.0, 1.0)[0]
    ax4.plot(new_FPRs, TPRs_predicted_fixed_3, label='fixed_threshold=0.7', linewidth=3, color='red', linestyle='dashed')

    ax4.plot([0,1], [0,1], linestyle='solid', linewidth=3, color='black')
    # ax4.plot([0,0], [0,1], linestyle='dashed')
    # ax4.plot([0,1], [0,0], linestyle='dashed')
    # plt.legend(loc='best')
    ax4.set_ylabel('True Positive rate (TPR)', fontsize=20)
    ax4.set_xlabel('False Positive rate (FPR)', fontsize=20)
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=4)

    print('AUC_adaptive:{}'.format(auc_adaptive))
    print('AUC_fixed1:{}'.format(auc_fixed1))
    print('AUC_fixed2:{}'.format(auc_fixed2))
    print('AUC_fixed3:{}'.format(auc_fixed3))

    # fig4, ax4 = plt.subplots()
    # ax4.scatter(adaptive_recalls, adaptive_precisions)
    # ax4.set_ylim([0,1])
    # ax4.set_xlim([0,1])
    # mymodel2 = np.poly1d(np.polyfit(adaptive_recalls, adaptive_precisions, 2))
    # new_recalls = np.linspace(0, 1, 200)
    # precisions_predicted = mymodel2(new_recalls)
    # ax4.plot(new_recalls, precisions_predicted)
    # ax4.plot([0,1], [0.5,0.5], linestyle='dashed')
    




    # fig1, ax1 = plt.subplots()
    # ax1.scatter(people_count, adaptive_thresholds, s=50, label='adaptive_threshold', color='blue')
    # ax1.plot(people_count, cross_means, label='cross-mean', linestyle='dashed', color='magenta')
    # ax1.plot(people_count, auto_means, label='auto-mean', linestyle='dashed', color='darkblue')
    # ax1.fill_between(people_count, cross_means, auto_means, color='deepskyblue', alpha=0.2)
    # ax2 = ax1.twinx()
    # ax2.plot(people_count, VALs, label='VAL', color='green')
    # ax2.plot(people_count, FARs, label='FAR', color='red')
    # plt.legend(loc='best')
    # fig1.tight_layout()

    fig5, ax51 = plt.subplots(num='adaptive-metrics-comparision')
    ax51.scatter(people_count, adaptive_thresholds, s=100, label='adaptive_threshold', color='blue')
    ax51.plot(people_count, cross_means, label='cross-mean', linestyle='dotted', color='magenta', linewidth=3)
    ax51.plot(people_count, auto_means, label='auto-mean', linestyle='dotted', color='indigo', linewidth=3)
    ax51.fill_between(people_count, cross_means, auto_means, color='deepskyblue', alpha=0.1)
    ax51.set_ylabel('Threshold', fontsize=20)
    ax51.set_xlabel('No. of Identities', fontsize=20)
    # ax51.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=6)
    ax52 = ax51.twinx()
    ax52.plot(people_count, adaptive_precisions, label='precision', color='blueviolet', linewidth=3)
    ax52.plot(people_count, adaptive_recalls, label='recall', color='green', linewidth=3)
    ax52.plot(people_count, adaptive_f1scores, label='f1-score', color='red', linewidth=3)
    ax52.set_ylabel('Performance Metrics', fontsize=20)
    ax52.set_xlabel('No. of Identities', fontsize=20)
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=6)
    h51, l51 = ax51.get_legend_handles_labels()
    h52, l52 = ax52.get_legend_handles_labels()
    ax51.legend(h51+h52, l51+l52, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=6)

    fig6, ax61 = plt.subplots(num='comparative-study-threshold-vs-f1score')
    ax61.scatter(people_count, adaptive_thresholds, s=100, label='adaptive', color='blue')
    ax61.scatter(people_count, fixed_thresholds_1, s=100, label='fixed@0.3', color='darkorange', marker='s')
    ax61.scatter(people_count, fixed_thresholds_2, s=100, label='fixed@0.5', color='green', marker='*')
    ax61.scatter(people_count, fixed_thresholds_3, s=100, label='fixed@0.7', color='red', marker='^')
    ax61.plot(people_count, cross_means, label='cross-mean', linestyle='dotted', color='magenta', linewidth=3)
    ax61.plot(people_count, auto_means, label='auto-mean', linestyle='dotted', color='indigo', linewidth=3)
    ax61.fill_between(people_count, cross_means, auto_means, color='deepskyblue', alpha=0.1)
    ax61.set_ylabel('Threshold', fontsize=20)
    ax61.set_xlabel('No. of Identities', fontsize=20)
    ax62 = ax61.twinx()
    # ax2.plot(people_count, adaptive_precisions, label='precision@adaptive_threshold', color='green')
    # ax2.plot(people_count, fixed_precisions_1, label='precision@fixed_threshold_1', color='red', linestyle='dashed')
    # ax2.plot(people_count, fixed_precisions_2, label='precision@fixed_threshold_2', color='green', linestyle='dashed')
    # ax2.plot(people_count, fixed_precisions_3, label='precision@fixed_threshold_3', color='blue', linestyle='dashed')
    ax62.plot(people_count, fixed_f1scores_1, label='f1-score@fixed=0.3',linestyle='dashed', color='darkorange', linewidth=3)
    ax62.plot(people_count, fixed_f1scores_2, label='f1-score@fixed=0.5', linestyle='dashed', color='green', linewidth=3)
    ax62.plot(people_count, fixed_f1scores_3, label='f1-score@fixed=0.7', linestyle='dashed', color='red', linewidth=3)
    ax62.plot(people_count, adaptive_f1scores, label='f1-score@adaptive', linewidth=3, color='blue')
    ax62.set_ylabel('F1-score', fontsize=20)
    ax62.set_xlabel('No. of Identities', fontsize=20)
    xnew = np.linspace(min(people_count), max(people_count), 100)
    spl_precision = make_interp_spline(people_count, adaptive_precisions)
    spl_recall = make_interp_spline(people_count, adaptive_recalls)
    spl_f1score = make_interp_spline(people_count, adaptive_f1scores)
    precision_new = spl_precision(xnew)
    recall_new = spl_recall(xnew)
    f1score_new = spl_f1score(xnew)
    # plt.legend(loc='best')
    # # fig2.tight_layout()
    h61, l61 = ax61.get_legend_handles_labels()
    h62, l62 = ax62.get_legend_handles_labels()
    ax61.legend(h61+h62, l61+l62, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=5)

    fig7, ax7 = plt.subplots(num='comparative-study-precision')
    ax7.plot(people_count, adaptive_precisions, label='adaptive_threshold', linewidth=3, color='blue')
    ax7.plot(people_count, fixed_precisions_1, label='fixed_threshold=0.3', linewidth=3, color='darkorange', linestyle='dashed')
    ax7.plot(people_count, fixed_precisions_2, label='fixed_threshold=0.5', linewidth=3, color='green', linestyle='dashed')
    ax7.plot(people_count, fixed_precisions_3, label='fixed_threshold=0.7', linewidth=3, color='red', linestyle='dashed')
    ax7.set_ylabel('Precision', fontsize=20)
    ax7.set_xlabel('No. of Identities', fontsize=20)
    # ax7.set_title('Comparative study of precision', fontsize=30)
    ax7.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=4)
    # fig6.tight_layout()

    fig8, ax8 = plt.subplots(num='comparative-study-recall')
    ax8.plot(people_count, adaptive_recalls, label='adaptive_threshold', linewidth=3, color='blue')
    ax8.plot(people_count, fixed_recalls_1, label='fixed_threshold=0.3', linewidth=3, color='darkorange', linestyle='dashed')
    ax8.plot(people_count, fixed_recalls_2, label='fixed_threshold=0.5', linewidth=3, color='green', linestyle='dashed')
    ax8.plot(people_count, fixed_recalls_3, label='fixed_threshold=0.7', linewidth=3, color='red', linestyle='dashed')
    # plt.legend(loc='upper center')
    ax8.set_ylabel('Recall', fontsize=20)
    ax8.set_xlabel('No. of Identities', fontsize=20)
    ax8.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=4)

    fig9, ax9 = plt.subplots(num='comparative-study-f1score')
    ax9.plot(people_count, adaptive_f1scores, label='adaptive_threshold', linewidth=3, color='blue')
    ax9.plot(people_count, fixed_f1scores_1, label='fixed_threshold=0.3', linewidth=3, color='darkorange', linestyle='dashed')
    ax9.plot(people_count, fixed_f1scores_2, label='fixed_threshold=0.5', linewidth=3, color='green', linestyle='dashed')
    ax9.plot(people_count, fixed_f1scores_3, label='fixed_threshold=0.7', linewidth=3, color='red', linestyle='dashed')
    ax9.set_ylabel('f1-score', fontsize=20)
    ax9.set_xlabel('No. of Identities', fontsize=20)
    # plt.legend(loc='upper center')
    # plt.title('Comparative study of f1-score')
    ax9.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=4)

    plt.show()

        # for i, current_encoding, facebox in zip(np.arange(len(query_embeddings)).tolist(), query_embeddings, face_boxes):
        #     dist_updated = list()
        #     for j, dict_name, dict_encodings in zip(np.arange(num_of_identities), known_face_names, known_face_encodings):
        #         dist = list()
        #         for dict_encoding in dict_encodings:
        #             # dist.append(euclidean_distance(dict_encoding, current_encoding))
        #             dist.append(cosine_similarity(dict_encoding, current_encoding))
        #         # eucliDist_matrix[i][j] = np.min(np.asarray(dist, dtype=np.float32))
                
        #         dist_updated.append(max(dist))

        #     if max(dist_updated) > 0.7:
        #         name = known_face_names[np.argmax(dist_updated)]
        #     else:
        #         name = 'Unknown'

if __name__ == '__main__':
    main()