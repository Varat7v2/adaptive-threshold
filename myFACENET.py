# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
from facenet import facenet, detect_face
import os, sys, glob
import time
import pickle

# modeldir = './model/facenet-20180408-102900.pb'     #output: 512D vector
modeldir = './model/facenet-20170511-185253.pb'   #output: 128D vector
mtcnn_models='./mtcnn-models'

class FACENET_EMBEDDINGS(object):
    def __init__(self, PATH_TO_CKPT):

        # self.detection_graph = tf.Graph()
        # with self.detection_graph.as_default():
        #     od_graph_def = tf.GraphDef()
        #     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        #         serialized_graph = fid.read()
        #         od_graph_def.ParseFromString(serialized_graph)
        #         tf.import_graph_def(od_graph_def, name='')


        # with self.detection_graph.as_default():
        #     config = tf.ConfigProto()
        #     config.gpu_options.allow_growth = True
        #     self.sess = tf.Session(graph=self.detection_graph, config=config)
        #     self.windowNotSet = True

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

            # with self.sess.as_default():
            facenet.load_model(PATH_TO_CKPT)
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = self.embeddings.get_shape()[1]
            self.emb_array = np.zeros((1, embedding_size))


    def run(self, faces):
        t1 = time.time()
        feed_dict = {self.images_placeholder: faces, self.phase_train_placeholder: False}
        self.emb_array = self.sess.run(self.embeddings, feed_dict=feed_dict)
        time_lapse = (time.time() - t1)*1000
        # print(str(time_lapse)+'milliseconds')

        return self.emb_array

    # with tf.Graph().as_default():
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    #     with sess.as_default():
    #         # print('Loading Modal')
    #         facenet.load_model(modeldir)
    #         images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    #         embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    #         phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    #         embedding_size = embeddings.get_shape()[1]
    #         emb_array = np.zeros((1, embedding_size))
                
    #         feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
    #         emb_array = sess.run(embeddings, feed_dict=feed_dict)
    
    # return emb_array
                    

def main():
    faces = list()
    myfacenet = FACENET_EMBEDDINGS(modeldir)
    for img in glob.glob('cropped_images'+'/*.jpg'):
        frame = cv2.resize(cv2.imread(img), (160,160), interpolation=cv2.INTER_LINEAR)
        faces.append(facenet.prewhiten(frame))
    
    face_encoding = myfacenet.run(faces)
    # print(face_encoding)

if __name__ == '__main__':
    main()