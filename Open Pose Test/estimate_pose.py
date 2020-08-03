import cv2
import numpy as np 
import argparse 
import time 

# parse commandline arguments
parser = argparse.ArgumentParser(description='This script demonstrates human pose estimation usin OpenPose CNNs with a pretrained model')
parser.add_argument('--proto', help='Path to .prototxt')
parser.add_argument('--model', help='Path to .caffeemodel')
parser.add_argument('--dataset', help='Kind of dataset the model was trained (COCO, MPI)')

args = parser.parse_args()

# prepare classes depending on dataset
if args.dataset == 'COCO':
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
elif args.dataset == 'MPI':
    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                   "Background": 15 }

    POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                   ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                   ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
else:
    BODY_PARTS ={"Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,"LShoulder":5,"LElbow":6,"LWrist":7,"MidHip":8,"RHip":9,"RKnee":10,"RAnkle":11,"LHip":12,"LKnee":13,"LAnkle":14,"REye":15,"LEye":16,"REar":17,"LEar":18,"LBigToe":19,"LSmallToe":20,"LHeel":21,"RBigToe":22,"RSmallToe":23,"RHeel":24,"Background":25}

    POSE_PAIRS =[ ["Neck","MidHip"],   ["Neck","RShoulder"],   ["Neck","LShoulder"],   ["RShoulder","RElbow"],   ["RElbow","RWrist"],   ["LShoulder","LElbow"],   ["LElbow","LWrist"],   ["MidHip","RHip"],   ["RHip","RKnee"],  ["RKnee","RAnkle"], ["MidHip","LHip"],  ["LHip","LKnee"], ["LKnee","LAnkle"],  ["Neck","Nose"],   ["Nose","REye"], ["REye","REar"],  ["Nose","LEye"], ["LEye","LEar"],   
["RShoulder","REar"],  ["LShoulder","LEar"],   ["LAnkle","LBigToe"],["LBigToe","LSmallToe"],["LAnkle","LHeel"], ["RAnkle","RBigToe"],["RBigToe","RSmallToe"],["RAnkle","RHeel"] ]

# scale image to defined proportions
WIDTH = 368
HEIGHT = 368

# read network graph and load pretrained weights on network
net = cv2.dnn.readNetFromCaffe(args.proto, args.model)

# initialize webcam
cam = cv2.VideoCapture(0)

while True:

    # cancel on ESC
    if cv2.waitKey(1) == 27:
        break

    # read image
    ret, frame = cam.read()

    cv2.namedWindow("Detection", cv2.WINDOW_AUTOSIZE)

    frameWith = frame.shape[0]
    frameHeight = frame.shape[1]

    # load image in a batch 
    inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (WIDTH, HEIGHT), (0, 0, 0), swapRB=False, crop=False)

    # set input for the net and start processing
    net.setInput(inp)

    startTime = time.time()
    out = net.forward()

    print("Processing time: ", time.time() - startTime)

    # show image and heatmap 
    cv2.namedWindow("Detection", cv2.WINDOW_AUTOSIZE)

    # grab the resulting points
    points = []

    for i in range(len(BODY_PARTS)):

        # get body part from result 
        heatMap = out[0, i, :, :]

        # for single human detection: get GLOBAL max location 
        _, conf, _, point = cv2.minMaxLoc(heatMap)

        x = (frameWith * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # add point if below confidence threshold
        points.append((int(x), int(y)) if conf > 0.1 else None)


    # draw points
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (255, 74, 0), 3)
            cv2.ellipse(frame, points[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)

    cv2.imshow("Detection", frame)
    
cv2.destroyAllWindows()
