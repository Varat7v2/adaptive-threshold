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

from scipy.optimize import minimize, minimize_scalar
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from scipy.interpolate import make_interp_spline, BSpline, CubicSpline
import scipy.integrate as integrate

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from facenet import facenet, detect_face
import adaptive_config as myconfig

class ADAPTIVE_THRESHOLD:
	def __init__(self, tDetector, myfacenet, myconfig):
		self.tDetector = tDetector
		self.myfacenet = myfacenet
		self.myconfig  = myconfig 

	def serialize_dict(self, dict_, filename):
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

	def deserialize_dict(self, filename):
	    # Deserialize dictionary
	    with open(filename, 'rb') as handle:
	        return pickle.load(handle)

	def face_distance(self, face_encoding, face_to_compare):
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


	def compare_faces(self, known_face_encodings, face_encoding_to_check, tolerance=0.9):
	    """
	    Compare a list of face encodings against a candidate encoding to see if they match.

	    :param known_face_encodings: A list of known face encodings
	    :param face_encoding_to_check: A single face encoding to compare against the list
	    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
	    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
	    """
	    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

	def cosine_similarity(self, dict_encoding, current_encoding):
	    return (np.dot(dict_encoding, current_encoding)/(np.linalg.norm(dict_encoding)*np.linalg.norm(current_encoding)))

	def cosine_distance(self, dict_encoding, current_encoding):
	    return (1 - np.dot(dict_encoding, current_encoding)/(np.linalg.norm(dict_encoding)*np.linalg.norm(current_encoding)))

	def euclidean_distance(self, dict_encoding, current_encoding):
	    return (np.sqrt(np.sum((dict_encoding - current_encoding) ** 2)))

	def facenet_encoding(self):
	    pass

	def dlib_encoding(self):
	    pass

	def get_facial_embeddings(self, cropped_faces):
	        return self.myfacenet.run(cropped_faces)

	def detect_crop_face(self, frame, frame_width, frame_height):
	    boxes, scores, classes, num_detections = self.tDetector.run(frame)
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
	                if self.myconfig.FACE_RECOGNITION == 'DLIB':
	                    face_encoding = FR.face_encoding(frame, face_locations)
	                if self.myconfig.FACE_RECOGNITION == 'FACENET':
	                #***********************************************************************************************
	                    current_encoding = self.myfacenet.run(faces_cropped)

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

	
	def evaluate_model(self, query_dict, gallery_dict, adaptive_threshold, P_same, P_diff):
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
	                    dist.append(self.cosine_similarity(q_embedding, g_embedding))

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
	    