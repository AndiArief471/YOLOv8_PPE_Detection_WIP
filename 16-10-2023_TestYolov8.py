from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

list = [200, 200, 400, 400]

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()

    result = model.predict(frame, show=True, classes=0)

    for r in result:
        boxes = r.boxes

        x1 = int(r[0].boxes.xyxy[0][0])
        y1 = int(r[0].boxes.xyxy[0][1])
        x2 = int(r[0].boxes.xyxy[0][2])
        y2 = int(r[0].boxes.xyxy[0][3])
        print(f"{x1}, {y1}, {x2}, {y2}")


    if (cv2.waitKey(30) == 32):
        cropped_frame = frame[y1:y2, x1:x2]
        result2 = model.predict(cropped_frame, classes=67)

    if (cv2.waitKey(30) == 27):
        break
