from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()

    list = []
    masks = []
    total_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    outside_mask = 255

    result = model.predict(frame, show=True, classes=0)
    for r in result:
        for i in range(len(r)):
            list.append(np.array(r[i].boxes.xyxy[0]))

    if (cv2.waitKey(30) == 32):
        print("Spacebar Pressed!")
        black_background = np.zeros_like(frame)
        for i in range(len(list)):
            print("Entering for loop")
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            masks.append(mask)
            cv2.rectangle(masks[i], (int(list[i][0]), int(list[i][1])), (int(list[i][2]), int(list[i][3])), 255, -1)
            outside_mask -= masks[i]
            total_mask += masks[i]
        print(outside_mask)
        print(total_mask)
        result2 = cv2.add(cv2.bitwise_and(frame, frame, mask=total_mask),
                          cv2.bitwise_and(black_background, black_background, mask=outside_mask))
        cv2.imshow('Image with Shapes on Black Background', result2)

    if (cv2.waitKey(1) == 27):
        break
