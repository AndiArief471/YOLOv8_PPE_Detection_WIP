from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv

#Object detection model & webcam
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

color = (0, 0, 255)
names = model.names

#Supervision bounding box
box_annotator = sv.BoxAnnotator(thickness=2,
                                text_thickness=0,
                                text_scale=0,)
#Main loop
while True:
    #Get each frame from webcam
    ret, frame = cap.read()

    #Reset variable each loop
    person = []
    masks = []
    total_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    outside_mask = 255

    #Predict only person
    result = model.predict(frame, classes=0, show_labels=False)
    #Get bounding box of each detected object
    for r in result:
        for i in range(len(r)):
            person.append(np.array(r[i].boxes.xyxy[0]))

    black_background = np.zeros_like(frame)

    #Make area outside of detected object black
    for i in range(len(person)):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        masks.append(mask)
        cv2.rectangle(masks[i], (int(person[i][0]), int(person[i][1])), (int(person[i][2]), int(person[i][3])), 255, -1)
        result2 = cv2.add(cv2.bitwise_and(frame, frame, mask=masks[i]),
                          cv2.bitwise_and(black_background, black_background, mask=255 - masks[i]))

        #Second prediction to detect PPE
        detect2 = model.predict(result2, classes=[67, 39], show_labels=False)
        PPE = []
        for d in detect2:
            for j in d.boxes.cls:
                PPE.append(names[int(j)])

        #Change the color of person[i] bounding box
        if PPE.count("cell phone") == 1 and PPE.count("bottle") == 1:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        #Draw person[i] bounding box
        cv2.rectangle(frame, (int(person[i][0]), int(person[i][1])), (int(person[i][2]), int(person[i][3])), color, 2)

    #Show the frame
    cv2.imshow('Image with Shapes on Black Background', frame)

    if (cv2.waitKey(1) == 27):
        break
