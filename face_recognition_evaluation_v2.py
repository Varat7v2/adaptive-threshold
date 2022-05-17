import cv2
import glob
import numpy as np
import pickle
import os, sys
import time
import pandas as pd
import matplotlib.pyplot as plt

from facenet import facenet, detect_face
from myFACENET_TFLITE import FACENET_TFLITE


faceDetection_frozenGraph import TensoflowFaceDector
from myFACENET import FACENET_EMBEDDINGS
from myMOBILEFACENET import MOBILEFACENET_EMBEDDINGS
import myface_recognition_config as myconfig

from sklearn import metrics
from tqdm import tqdm
from collections import defaultdict

from myDLIB import DLIB_FACERECO

def serialize_dict(dict_, filename):
    mydict = dict()
    # Load the embeddings of known people or check if there are any new people need to add in the list
    for image in glob.glob('my_known_persons/*'):
        name = image.split('/')[-1].split('.')[0]
        mydict[name] = encoding
    # Serialize dictionary in binary format
    with open(filename, 'wb') as handle:
        pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)

def deserialize_dict(filename):
    # Deserialize dictionary
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, ord=None, axis=1)


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


def main():
    face_detector = TensoflowFaceDector(myconfig.FACE_DETECTION_MODEL_512)
    myfacenet = FACENET_EMBEDDINGS(myconfig.RECO_FACENET_128D)
    myfacenet_tflite = FACENET_TFLITE(myconfig.RECO_FACENET_TFLITE_FLOAT32)
    mymobilefacenet_pb = MOBILEFACENET_EMBEDDINGS(myconfig.RECO_MOBILEFACENET_128D)
    dlib_fr = DLIB_FACERECO(myconfig.RECO_DLIB_128D)

    ### READ CSV AND FACE-EMBEDDINGS FILE
    dict_known = deserialize_dict('data/evaluation/{}/{}.pkl'.format(myconfig.DATASET, myconfig.SERIALIZED_FILE))
    df_csv = pd.read_csv('data/evaluation/{}/images_{}.csv'.format(myconfig.DATASET, myconfig.SERIALIZED_FILE))

    known_face_names, known_face_encodings = list(), list()
    for key, values in dict_known.items():
        # print('Dictionary embedding sample size: ', len(values[0]))
        known_face_names.append(key)
        known_face_encodings.append(values)

    # for name, encodings in zip(known_face_names, known_face_encodings):
    #     print(name)
    #     print(len(encodings))
    #     for encoding in encodings:
    #         # print(encoding)
    #         pass
        
    num_of_identities = len(known_face_names)
    print('Number of Identities: {}'.format(num_of_identities))
    # print('Identities: {}'.format(known_face_names))
    print('Facial Embedding size: {}'.format(len(known_face_encodings[0][0])))

    count = 0
    wrongImg_count = 0
    y_true = list()
    y_pred = list()

    for fname, img in zip(df_csv['name'], df_csv['filename']):
        t1 = time.time()
        count += 1
        imgfile = img.split('/')[-1]

        frame = cv2.imread(img)
        frame_height, frame_width, _ = frame.shape
        
        boxes, scores, classes, num_detections = face_detector.run(frame)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        face_locations = list()
        cropped_faces = list()
        face_boxes = list()

        scores_updated = list(score for score in scores if score > myconfig.FACE_DETECTION_THRESHOLD)
        score_count = len(scores_updated)
        boxes_updated = list(boxes[i] for i in range(score_count))

        if len(boxes_updated) == 1:
            for score, box in zip(scores_updated, boxes_updated):
                # ymin, xmin, ymax, xmax = box
                left = int(box[1]*frame_width)
                top = int(box[0]*frame_height)
                right = int(box[3]*frame_width)
                bottom = int(box[2]*frame_height)

                face_locations.append((top, right, bottom, left))
                face_boxes.append([left, top, right, bottom])
                cropped_face = frame[top:bottom, left:right]
                
                if myconfig.FACE_RECOGNITION_MODEL == 'FACENET':
                    cropped_face = cv2.resize(cropped_face, (160,160), interpolation=cv2.INTER_CUBIC)
                    cropped_face = facenet.prewhiten(cropped_face)    # Normalization
                    cropped_faces.append(cropped_face)
                elif myconfig.FACE_RECOGNITION_MODEL == 'DLIB':
                    cropped_face = cv2.resize(cropped_face, (150, 150), interpolation=cv2.INTER_CUBIC)
                    cropped_faces.append(cropped_face)
                elif myconfig.FACE_RECOGNITION_MODEL == 'MOBILEFACENET':
                    frame1 = cv2.resize(cropped_face, (112,112), interpolation=cv2.INTER_LINEAR)
                    frame1 = np.asarray(frame1, dtype=np.float32)
                    frame1 = (frame1 - 127.5) / 128.0
                    frame1 = np.expand_dims(frame1, axis=0)
                    # frame2 = cv2.resize(cropped_face, (112,112), interpolation=cv2.INTER_LINEAR)
                    # frame2 = np.asarray(frame2, dtype=np.float32)
                    # frame2 = (frame2 - 127.5) / 128.0
                    # frame2 = np.expand_dims(frame2, axis=0)
                    cropped_faces = np.concatenate([frame1, frame1])
            
            num_curr_faces = len(cropped_faces)
            eucliDist_matrix = np.zeros((num_curr_faces, num_of_identities))

            # FACIAL EMBEDDING USING DLIB / FACENET MODEL
            if myconfig.FACE_RECOGNITION_MODEL == 'FACENET':
                current_encodings = myfacenet.run(cropped_faces)
            elif myconfig.FACE_RECOGNITION_MODEL == 'DLIB':
                current_encodings = dlib_fr.run(cropped_faces)
            elif myconfig.FACE_RECOGNITION_MODEL == 'MOBILEFACENET':
                current_encodings = mymobilefacenet_pb.run(cropped_faces)
                current_encodings = [current_encodings[0]]
            # elif myconfig.MODEL_TYPE == 'tflite':
            #     current_encodings = np.squeeze(myfacenet_tflite.run(cropped_faces))
            #     if current_encodings.ndim == 1:
            #         current_encodings = current_encodings.reshape((1, -1))

            for i, current_encoding, facebox in zip(np.arange(len(current_encodings)).tolist(), current_encodings, face_boxes):
                dist_updated = list()
                for j, dict_name, dict_encodings in zip(np.arange(num_of_identities), known_face_names, known_face_encodings):
                    dist = list()
                    for dict_encoding in dict_encodings:
                        # dist.append(euclidean_distance(dict_encoding, current_encoding))
                        dist.append(cosine_similarity(dict_encoding, current_encoding))
                    eucliDist_matrix[i][j] = np.min(np.asarray(dist, dtype=np.float32))
                    
                    # dist_updated.append(max(dist)) # cosine similarity
                    dist_updated.append(max(dist)) # euclidean distance
                # print(min(dist))

                if max(dist_updated) > myconfig.SIMILARITY_THRESHOLD:
                    name = known_face_names[np.argmax(dist_updated)]
                else:
                    name = 'Unknown'

                y_true.append(True)

                if fname == name:
                    y_pred.append(True)
                else:
                    y_pred.append(False)    
        else:
            wrongImg_count += 1
            print('WARNINIG: More than 1 faces detected in IMAGE: {}!!!'.format(imgfile))

        # for index in np.argmin(eucliDist_matrix, axis=1): # Row: 1, Cols: 0
        #     # print(np.min(eucliDist_matrix, axis=1))
        #     if np.min(eucliDist_matrix, axis=1) < 0.8:
        #         print(known_face_names[index])
        #     else:
        #         print('Unknown')1

        time_lapse = time.time() - t1

    ### MODEL EVALUATION
    print('No. of test images: {}'.format(count))
    print('No. of wrong images during evaluation: {}'.format(wrongImg_count))
    print('y_pred length={}'.format(len(y_pred)))
    print('y_true length={}'.format(len(y_true)))
    print('Precision = {:.2f}'.format(metrics.precision_score(y_true, y_pred)))
    print('Recall = {:.2f}'.format(metrics.recall_score(y_true, y_pred)))
    print('F1-score = {:.2f}'.format(metrics.f1_score(y_true, y_pred)))
    # print('Average precision score = {:.2f}'.format(metrics.average_precision_score(y_true, y_pred)))
    print(metrics.classification_report(y_true, y_pred))
    print(metrics.confusion_matrix(y_true, y_pred, normalize='all'))
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    print('TP={}, FP={}, FN={}, TN={}'.format(tp, fp, fn, tn))

if __name__ == '__main__':
    main()