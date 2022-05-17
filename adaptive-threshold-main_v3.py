# import FaceRecognition as FR
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
from adaptive_threshold import ADAPTIVE_THRESHOLD
from threshold_statistics import THRESHOLD_STATISTICS

font = {'size': 10}
plt.rc('font', **font)

linearReg = LinearRegression()

sys.setrecursionlimit(10**7) 
print('Recusion Limit: {}'.format(sys.getrecursionlimit()))


OUTPUT_CSV_PATH = 'results_csv_{}_{}'.format(myconfig.GSIZE, myconfig.DATA_VERSION)
GALLERY_FVECTORS = myconfig.GSIZE

if not os.path.exists(OUTPUT_CSV_PATH):
    os.makedirs(OUTPUT_CSV_PATH)

### Class Objects Definition
tDetector   = TensoflowFaceDector(myconfig.PATH_TO_CKPT_FACE)
myfacenet   = FACENET_EMBEDDINGS(myconfig.PATH_TO_CKPT_FACENET_512D_9967)
aThreshold  = ADAPTIVE_THRESHOLD(tDetector, myfacenet, myconfig)
thresStats  = THRESHOLD_STATISTICS(aThreshold, myconfig)

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

def main():
    tstart = time.time()
    global query_dict, gallery_dict
    query_dict = dict()
    gallery_dict = dict()
    print('Creating gallery and query dictionary...')
    auto_means = list()
    cross_means = list()

    updated_dirs = list()
    directories = [mydir for mydir in os.listdir(myconfig.SRC_PATH)]
    FOLDER_COUNT = 0
    for dir_name in directories:
        images_count = len(list(f for f in glob.glob(os.path.join(myconfig.SRC_PATH, dir_name)+'/*')))
        if images_count >= myconfig.IMAGE_NUM_MIN and images_count <= myconfig.IMAGE_NUM_MAX:
            FOLDER_COUNT += 1
            updated_dirs.append(dir_name)

    # print('folder count: ', FOLDER_COUNT)
    # sys.exit(0)

    adaptive_precisions, adaptive_recalls, adaptive_f1scores, adaptive_thresholds = list(), list(), list(), list()
    fixed_thresholds_1, fixed_precisions_1, fixed_recalls_1, fixed_f1scores_1 = list(), list(), list(), list()
    fixed_thresholds_2, fixed_precisions_2, fixed_recalls_2, fixed_f1scores_2 = list(), list(), list(), list()
    fixed_thresholds_3, fixed_precisions_3, fixed_recalls_3, fixed_f1scores_3 = list(), list(), list(), list()

    # For error handling - manually adding few list values in the begining
    TPRs_adaptive, FPRs_adaptive = [0,0,0,0,0,0,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1]
    TPRs_fixed_1, FPRs_fixed_1   = [0,0,0,0,0,0,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1]
    TPRs_fixed_2, FPRs_fixed_2   = [0,0,0,0,0,0,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1]
    TPRs_fixed_3, FPRs_fixed_3   = [0,0,0,0,0,0,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1]


    loop_count = 0
    dir_count = 0
    
    
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
            myconfig.PLOT_FLAG = True
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
                        frame, faces_cropped, face_boxes, current_encoding = aThreshold.detect_crop_face(frame, 
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
            adaptive_threshold, auto_mean, cross_mean, P_same, P_diff = thresStats.find_threshold_stats(mydirs)
            # print(adaptive_threshold)
            # adaptive_thresholds.append(adaptive_threshold)
            auto_means.append(auto_mean)
            cross_means.append(cross_mean)
            # print('Adaptive threshold: {}'.format(adaptive_threshold))

            # # SERIALIZING GALLERY AND QUERY EMBEDDINGS
            # aThreshold.serialize_dict(gallery_dict, 'data/gallery_dict.pkl')
            # aThreshold.serialize_dict(query_dict, 'data/query_dict.pkl')

            # ADAPTIVE THRESHOLD
            adaptive_precision, adaptive_recall, adaptive_f1score, adaptive_accuracy, adaptive_tpr, adaptive_fpr = aThreshold.evaluate_model(
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
                adaptive_precision, adaptive_recall, adaptive_f1score, adaptive_accuracy, adaptive_tpr, adaptive_fpr = aThreshold.evaluate_model(
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
            precision_fixed_1, recall_fixed_1, f1score_fixed_1,accuracy_fixed1, tpr_fixed_1, fpr_fixed_1 = aThreshold.evaluate_model(
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
            precision_fixed_2, recall_fixed_2, f1score_fixed_2, accuracy_fixed2, tpr_fixed_2, fpr_fixed_2 = aThreshold.evaluate_model(
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
            precision_fixed_3, recall_fixed_3, f1score_fixed_3,accuracy_fixed3, tpr_fixed_3, fpr_fixed_3 = aThreshold.evaluate_model(
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

    #### PLOT - ROC CURVE -------------------------------------------------------------------------------
    fig4, ax4 = plt.subplots(num='roc_curve')
    ax4.scatter(FPRs_adaptive, TPRs_adaptive, s=50, color='blue')
    ax4.scatter(FPRs_fixed_1, TPRs_fixed_1, s=50, marker='s', color='darkorange')
    ax4.scatter(FPRs_fixed_2, TPRs_fixed_2, s=50, marker='*', color='green')
    ax4.scatter(FPRs_fixed_3, TPRs_fixed_3, s=50, marker='^', color='red')
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

    ax4.plot(new_FPRs, TPRs_predicted, label='adaptive_threshold', linewidth=2, color='blue')

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
    ax4.plot(new_FPRs, TPRs_predicted_fixed_1, label='fixed_threshold=0.3', linewidth=2, color='darkorange', linestyle='dashed')

    mypoly2 = np.polyfit(FPRs_fixed_2, TPRs_fixed_2, 2)
    mymodel2 = np.poly1d(mypoly2)
    TPRs_predicted_fixed_2 = mymodel2(new_FPRs)
    auc_fixed2 = integrate.quad(lambda x: myquadeqn(mypoly2, x), 0.0, 1.0)[0]
    ax4.plot(new_FPRs, TPRs_predicted_fixed_2, label='fixed_threshold=0.5', linewidth=2, color='green', linestyle='dashed')

    # print(FPRs_fixed_3)
    # print(TPRs_fixed_3)
    # sys.exit(0)
    
    mypoly3 = np.polyfit(FPRs_fixed_3, TPRs_fixed_3, 2)
    mymodel3 = np.poly1d(mypoly3)
    TPRs_predicted_fixed_3 = mymodel3(new_FPRs)
    auc_fixed3 = integrate.quad(lambda x: myquadeqn(mypoly3, x), 0.0, 1.0)[0]
    ax4.plot(new_FPRs, TPRs_predicted_fixed_3, label='fixed_threshold=0.7', linewidth=2, color='red', linestyle='dashed')

    ax4.plot([0,1], [0,1], linestyle='solid', linewidth=2, color='black')
    # ax4.plot([0,0], [0,1], linestyle='dashed')
    # ax4.plot([0,1], [0,0], linestyle='dashed')
    # plt.legend(loc='best')
    ax4.set_ylabel('True Positive rate (TPR)', fontsize=12)
    ax4.set_xlabel('False Positive rate (FPR)', fontsize=12)
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=False, ncol=4)
    fig4.tight_layout()
    fig4.savefig(myconfig.PLOT_PATH+'/roc_curve.pdf', bbox_inches='tight', dpi=300)

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


    # PLOT - ADAPTIVE METRICS COMPARISION -------------------------------------------------------------------
    fig5, ax51 = plt.subplots(num='adaptive-metrics-comparision', figsize=(9, 5))
    ax51.scatter(people_count, adaptive_thresholds, s=50, label='adaptive_threshold', color='blue')
    ax51.plot(people_count, cross_means, label='cross-mean', linestyle='dotted', color='magenta', linewidth=1.5)
    ax51.plot(people_count, auto_means, label='auto-mean', linestyle='dotted', color='indigo', linewidth=1.5)
    ax51.fill_between(people_count, cross_means, auto_means, color='deepskyblue', alpha=0.1)
    ax51.set_ylabel('Threshold', fontsize=10)
    ax51.set_xlabel('No. of Identities', fontsize=10)
    # ax51.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=6)
    ax52 = ax51.twinx()
    ax52.plot(people_count, adaptive_precisions, label='precision', color='blueviolet', linewidth=1.5)
    ax52.plot(people_count, adaptive_recalls, label='recall', color='green', linewidth=1.5)
    ax52.plot(people_count, adaptive_f1scores, label='f1-score', color='red', linewidth=1.5)
    ax52.set_ylabel('Performance Metrics', fontsize=10)
    ax52.set_xlabel('No. of Identities', fontsize=10)
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=6)
    h51, l51 = ax51.get_legend_handles_labels()
    h52, l52 = ax52.get_legend_handles_labels()
    ax51.legend(h51+h52, l51+l52, loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=False, ncol=6)
    # fig5.tight_layout()
    fig5.savefig(myconfig.PLOT_PATH+'/adaptive-metrics-comparision.pdf', dpi=300)

    fig6, ax61 = plt.subplots(num='comparative-study-threshold-vs-f1score')
    ax61.scatter(people_count, adaptive_thresholds, s=100, label='adaptive', color='blue')
    ax61.scatter(people_count, fixed_thresholds_1, s=100, label='fixed@0.3', color='darkorange', marker='s')
    ax61.scatter(people_count, fixed_thresholds_2, s=100, label='fixed@0.5', color='green', marker='*')
    ax61.scatter(people_count, fixed_thresholds_3, s=100, label='fixed@0.7', color='red', marker='^')
    ax61.plot(people_count, cross_means, label='cross-mean', linestyle='dotted', color='magenta', linewidth=2)
    ax61.plot(people_count, auto_means, label='auto-mean', linestyle='dotted', color='indigo', linewidth=2)
    ax61.fill_between(people_count, cross_means, auto_means, color='deepskyblue', alpha=0.1)
    ax61.set_ylabel('Threshold', fontsize=12)
    ax61.set_xlabel('No. of Identities', fontsize=12)
    ax62 = ax61.twinx()
    # ax2.plot(people_count, adaptive_precisions, label='precision@adaptive_threshold', color='green')
    # ax2.plot(people_count, fixed_precisions_1, label='precision@fixed_threshold_1', color='red', linestyle='dashed')
    # ax2.plot(people_count, fixed_precisions_2, label='precision@fixed_threshold_2', color='green', linestyle='dashed')
    # ax2.plot(people_count, fixed_precisions_3, label='precision@fixed_threshold_3', color='blue', linestyle='dashed')
    ax62.plot(people_count, fixed_f1scores_1, label='f1-score@fixed=0.3',linestyle='dashed', color='darkorange', linewidth=2)
    ax62.plot(people_count, fixed_f1scores_2, label='f1-score@fixed=0.5', linestyle='dashed', color='green', linewidth=2)
    ax62.plot(people_count, fixed_f1scores_3, label='f1-score@fixed=0.7', linestyle='dashed', color='red', linewidth=2)
    ax62.plot(people_count, adaptive_f1scores, label='f1-score@adaptive', linewidth=2, color='blue')
    ax62.set_ylabel('F1-score', fontsize=12)
    ax62.set_xlabel('No. of Identities', fontsize=12)
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
    ax61.legend(h61+h62, l61+l62, loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=False, ncol=5)
    fig6.savefig(myconfig.PLOT_PATH+'/comparative-study-threshold-vs-f1score.pdf', bbox_inches='tight', dpi=300)

    fig7, ax7 = plt.subplots(num='comparative-study-precision')
    ax7.plot(people_count, adaptive_precisions, label='adaptive_threshold', linewidth=2, color='blue')
    ax7.plot(people_count, fixed_precisions_1, label='fixed_threshold=0.3', linewidth=2, color='darkorange', linestyle='dashed')
    ax7.plot(people_count, fixed_precisions_2, label='fixed_threshold=0.5', linewidth=2, color='green', linestyle='dashed')
    ax7.plot(people_count, fixed_precisions_3, label='fixed_threshold=0.7', linewidth=2, color='red', linestyle='dashed')
    ax7.set_ylabel('Precision', fontsize=12)
    ax7.set_xlabel('No. of Identities', fontsize=12)
    # ax7.set_title('Comparative study of precision', fontsize=30)
    ax7.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=False, ncol=4)
    # fig6.tight_layout()
    fig7.savefig(myconfig.PLOT_PATH+'/comparative-study-precision.pdf', bbox_inches='tight', dpi=300)

    fig8, ax8 = plt.subplots(num='comparative-study-recall')
    ax8.plot(people_count, adaptive_recalls, label='adaptive_threshold', linewidth=2, color='blue')
    ax8.plot(people_count, fixed_recalls_1, label='fixed_threshold=0.3', linewidth=2, color='darkorange', linestyle='dashed')
    ax8.plot(people_count, fixed_recalls_2, label='fixed_threshold=0.5', linewidth=2, color='green', linestyle='dashed')
    ax8.plot(people_count, fixed_recalls_3, label='fixed_threshold=0.7', linewidth=2, color='red', linestyle='dashed')
    # plt.legend(loc='upper center')
    ax8.set_ylabel('Recall', fontsize=12)
    ax8.set_xlabel('No. of Identities', fontsize=12)
    ax8.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=False, ncol=4)
    fig8.savefig(myconfig.PLOT_PATH+'/comparative-study-recall.pdf', bbox_inches='tight', dpi=300)

    fig9, ax9 = plt.subplots(num='comparative-study-f1score')
    ax9.plot(people_count, adaptive_f1scores, label='adaptive_threshold', linewidth=2, color='blue')
    ax9.plot(people_count, fixed_f1scores_1, label='fixed_threshold=0.3', linewidth=2, color='darkorange', linestyle='dashed')
    ax9.plot(people_count, fixed_f1scores_2, label='fixed_threshold=0.5', linewidth=2, color='green', linestyle='dashed')
    ax9.plot(people_count, fixed_f1scores_3, label='fixed_threshold=0.7', linewidth=2, color='red', linestyle='dashed')
    ax9.set_ylabel('f1-score', fontsize=12)
    ax9.set_xlabel('No. of Identities', fontsize=12)
    # plt.legend(loc='upper center')
    # plt.title('Comparative study of f1-score')
    ax9.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=False, ncol=4)
    fig9.savefig(myconfig.PLOT_PATH+'/comparative-study-f1score.pdf', bbox_inches='tight', dpi=300)

    plt.show()
    tend = time.time()
    print('The total processing time:', (tend-tstart)/(60*60), 'hours.')

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