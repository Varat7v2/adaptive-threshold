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

class THRESHOLD_STATISTICS():
    def __init__(self, aThreshold, myconfig):
        self.aThreshold = aThreshold
        self.myconfig   = myconfig 

    def find_threshold_stats(self, mydirs):
        mydict_list = list()
        # print(mydirs)
        for dir_name in mydirs:
            images_count = len(list(f for f in glob.glob(os.path.join(self.myconfig.SRC_PATH, dir_name)+'/*')))
            if images_count >= self.myconfig.IMAGE_NUM_MIN and images_count <= self.myconfig.IMAGE_NUM_MAX:
                cropped_faces = list()
                faces_embeddings = list()
                mydict = dict()
                count = 0
                for file in glob.glob(os.path.join(self.myconfig.SRC_PATH, dir_name)+'/*'):
                    filename = file.split('/')[-1]
                    frame = cv2.imread(file)
                    frame_height, frame_width, _ = frame.shape
                    #check if there are two faces in the frame
                    frame, cropped, face_boxes, current_encoding = self.aThreshold.detect_crop_face(frame, frame_width, frame_height)
                    if cropped is not None and current_encoding is not None:
                        cropped_faces.append(cropped)
                        faces_embeddings.append(current_encoding)

                    count += 1


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
                for k, (dict_encoding) in enumerate(zip(dict_encodings)):                   #each image for that person
                    dict_encoding = np.squeeze(np.asarray(dict_encoding), axis=0)
                    for l, (dict_encoding_copy) in enumerate(zip(dict_encodings_copy)):    #each images of gallery
                        dict_encoding_copy = np.squeeze(np.asarray(dict_encoding_copy), axis=0).transpose()
                        if i == j:
                            if k != l:
                                P_same += 1
                                auto_similarity.append(self.aThreshold.cosine_similarity(dict_encoding, dict_encoding_copy).tolist()[0][0])
                        else:
                            P_diff += 1
                            cross_similarity.append(self.aThreshold.cosine_similarity(dict_encoding, dict_encoding_copy).tolist()[0][0])

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

        if self.myconfig.PLOT_FLAG:
            # Density Plot and Histogram of all arrival delays
            fig1 = plt.figure(num='histogram-gaussian-curve')
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
            ax1.plot(auto_x, auto_y, color='blue', linewidth=2, label='auto_gaussian')

            cross_x = np.linspace(cross_min-0.1, cross_max+0.2, 1000)
            cross_y = scipy.stats.norm.pdf(cross_x, cross_mean, cross_std)
            # cross_ymin = np.min(cross_y)
            # cross_ymax = np.max(cross_y)
            # cross_y = (cross_y-cross_ymin)/(cross_ymax-cross_ymin)
            ax1.plot(cross_x, cross_y, color='red', linewidth=2, label='cross_gaussian')
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=False, ncol=4)
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
            
            # fig1.subplots_adjust(bottom=0.2)
            fig1.tight_layout()
            fig1.savefig(self.myconfig.PLOT_PATH+'/histogram-gaussian-curve.pdf', bbox_inches='tight', dpi=300)
            # sys.exit(0)

            fig2 = plt.figure(num='auto-similarity distribution')
            plt.scatter(np.arange(len(auto_similarity)), auto_similarity, s=1, label='auto_similarity', color='deepskyblue')
            plt.plot(np.arange(len(auto_similarity)), [auto_avg]*len(auto_similarity), label='auto_average', linewidth=1, color='green')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=False, ncol=2)
            plt.xlabel('No. of similarity pairing')
            plt.ylabel('Cosine similarity')
            # plt.legend(loc="best")
            # auto_dist_1 = np.asarray(auto_dist_1)
            fig2.tight_layout()
            fig2.savefig(self.myconfig.PLOT_PATH+'/auto-similarity distribution.pdf', bbox_inches='tight', dpi=300)

            fig3 = plt.figure(num='cross-similarity distribution')
            plt.scatter(np.arange(len(cross_similarity)), cross_similarity, s=1, label='cross_similarity', color='coral')
            plt.plot(np.arange(len(cross_similarity)), [cross_avg]*len(cross_similarity), label='cross_average', linewidth=1, color='green')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=False, ncol=2)
            plt.xlabel('No. of similarity pairing')
            plt.ylabel('Cosine similarity')
            # plt.legend(loc="best")
            # cross_dist_1 = np.asarray(cross_dist_1)
            fig3.tight_layout()
            fig3.savefig(self.myconfig.PLOT_PATH+'/cross-similarity distribution.pdf', bbox_inches='tight', dpi=300)

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
            ax10_2.set_ylabel('Auto-similarity distribution', fontsize=10)
            ax10_2.set_xlabel('Cosine similarity', fontsize=10)
            # plt.figure('Cross similarity histogram')
            ax10_1.hist(cross_similarity, bins, alpha=0.8, label='cross_similarity', color='coral')
            ax10_1.set_ylabel('Cross-similarity distribution', fontsize=10)
            ax10_1.set_xlabel('Cosine similarity', fontsize=10)
            # plt.xlabel('Cosine similarity')
            # plt.ylabel('No. of occurances')
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=False, ncol=2)
            h10_1, l10_1 = ax10_1.get_legend_handles_labels()
            h10_2, l10_2 = ax10_2.get_legend_handles_labels()
            ax10_2.legend(h10_1+h10_2, l10_1+l10_2, loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=False, ncol=6)
            fig10.savefig(self.myconfig.PLOT_PATH+'/Histogram.pdf', bbox_inches='tight', dpi=300)

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