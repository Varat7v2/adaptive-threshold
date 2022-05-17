 
# Flag to decide to store image directly form webcam
ADD_PERSONS = False

PLOT_FLAG = False

# Dataset chosen for calculating the adaptive threshold
# SRC_PATH = 'data/data_athelets'
SRC_PATH = 'data/lfw'
PLOT_PATH = 'plots'

# Type of source: Either IMAGE or WEBCAM
SOURCE = 'image'
IMG_WIDTH = 1200
IMG_HEIGHT = 900

# Limiting the number of images per identity
#  Considering identity only with a single image does not help
IMAGE_NUM_MIN = 4 #4
IMAGE_NUM_MAX = 60 #60

# Limit to measure model accuracy constrained to f1-score/precision greater than equal to 80%
METRIC_BOUND = 0.8

# Gallery size (1:storing only a single image of a identity in the database)
# Higher number of gallery images will increase scope of face-orientation but computationally expensive
GSIZE=1

#number of folders to select
FOLDER_PERCENT=1.0

#number of images to choose from selected folders
IMAGE_PERCENT=1.0

# Naming the folder in which all the results output will be stored. It can be named anything
DATA_VERSION=1

# Face recognition model
FACE_RECOGNITION = 'FACENET'
#FACE_RECOGNITION = 'DLIB'

# Face detetection model
PATH_TO_CKPT_FACE = 'models/face_ssd_512x512.pb'

# Facenet models for extracting 128-dimensional facial feature vectors
PATH_TO_CKPT_FACENET_128D = 'models/facenet-20170511-185253.pb'

# Facenet models for extracting 512-dimensional facial feature vectors
PATH_TO_CKPT_FACENET_512D_9905 = 'models/facenet-20180408-102900-CASIA-WebFace.pb'
PATH_TO_CKPT_FACENET_512D_9967 = 'models/faenet-20180402-114759-VGGFace2.pb'

## GRAPH CONFIGURATION


## OTHERS CONFIGURATION
PRINT_IDENTITIES_NUM = False