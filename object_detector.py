from imutils import paths
import numpy as np
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default = 0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

prototxt = args["prototxt"]
caffeModel = args["model"]
imagePath = args["image"]

imageWidth = 300
imageHeight = 300

# Initialize the class labels for MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "dining table",
	"dog", "horse", "motorbike", "person", "pot plant", "sheep",
	"sofa", "train", "tv monitor"]
# Generate colors for the bounding boxes for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))