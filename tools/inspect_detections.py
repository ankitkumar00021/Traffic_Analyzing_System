from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('dashcam.mp4')
labels_seen = {}
frames=0
while frames<10:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, imgsz=640, conf=0.25, verbose=False)
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            name = model.names[int(cls)]
            labels_seen[name] = labels_seen.get(name,0)+1
            print(f'Frame {frames+1}: {name} ({float(conf):.2f})')
    frames+=1
cap.release()
print('Summary labels:', labels_seen)
