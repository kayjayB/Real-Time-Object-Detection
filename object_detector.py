from imutils import paths
import numpy as np
import argparse
import cv2

def extract(detections, height, width):
    confidences = detections[0,0,:,2] # (1,1,n,2) = Confidence
    classes = detections[0,0,:,1] # (1,1,n,1) = Class
    # (1,1,n,3:7) = Bounding box
    boxes = detections[0, 0, :, 3:7] * np.array([width, height, width, height])
    return confidences, classes.astype('int'), boxes.astype('int')

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
confidenceLevel = args["confidence"]

imageWidth = 300
imageHeight = 300

# Initialize the class labels for MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "dining table",
	"dog", "horse", "motorbike", "person", "pot plant", "sheep",
	"sofa", "train", "tv monitor"]
# Generate colors for the bounding boxes for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO]: Loading pretrained network....")
network = cv2.dnn.readNetFromCaffe(prototxt, caffeModel)

image = cv2.imread(imagePath)
image = cv2.resize(image, (imageWidth,imageHeight))
blob = cv2.dnn.blobFromImage(image, 0.007843, (imageWidth,imageHeight),127.5)

# Set the image as the input to the network
network.setInput(blob)
detections = network.forward() # (1,1,n,7) numpy n-dimensional array

for i in np.arange(0, detections.shape[2]):
    confidences, classes, boxes = extract(detections, imageHeight, imageWidth)

    if confidences[i] > confidenceLevel:
        classIndex = classes[i]
        (startX, startY, endX, endY) = boxes[i][0:4]

        # Display the box on the image
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[classIndex], 2)

        # Display the label and the confidence on the image
        label = "{}: {:.2f}%".format(CLASSES[classIndex], confidences[i] * 100)
        print("[INFO] {}".format(label))
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[classIndex], 2)

cv2.imshow("Detected Objects", image)
cv2.waitKey(0)


