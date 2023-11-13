from ultralytics import YOLO
import cv2
import numpy as np

def click_event(event, x, y, flags, params):
    global clicked_points
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # appending the clicked point to the list
        clicked_points.append((x, y))

#Object detection model & webcam
model = YOLO('Person.pt')
model2 = YOLO('APD.pt')
cap = cv2.VideoCapture('APD Project/CCTV_FAD_output2.mp4')
cap.set(3, 1280)
cap.set(4, 720)

color = (0, 0, 255)
names = model2.names

clicked_points = []
line_visible = False
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
        detect2 = model2.predict(result2, show_labels=False)
        PPE = []
        for d in detect2:
            for j in d.boxes.cls:
                PPE.append(names[int(j)])

        if "No Helmet" in PPE:
            color = (0, 0, 255)
        elif "No Glove" in PPE:
            color = (0, 255, 255)
        elif "Helmet" in PPE and "Glove" in PPE and "Boot" in PPE:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        #Change the color of person[i] bounding box
        print(f"detected PPE = {PPE}")
        print(color)
        # if PPE.count("Helmet") > 0:
        #     color = (0, 255, 255)
        #     if PPE.count("Boot") > 0:
        #         color = (0, 255, 0)
        # else:
        #     color = (0, 0, 255)

        #Draw person[i] bounding box
        cv2.rectangle(frame, (int(person[i][0]), int(person[i][1])), (int(person[i][2]), int(person[i][3])), color, 2)

    for point in clicked_points:
        cv2.circle(frame, point, 5, (255, 0, 0), -1)

    if line_visible == True:
        pts = np.array(clicked_points, np.int32)
        cv2.polylines(frame, [pts], True, (255, 0, 0))
    else:
        if (cv2.waitKey(30) == 32):
            line_visible = True
    #Show the frame
    cv2.imshow('Image with Shapes on Black Background', frame)

    cv2.setMouseCallback('Image with Shapes on Black Background', click_event)
    print(clicked_points)

    if (cv2.waitKey(1) == 27):
        break