import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

cap = cv2.VideoCapture(0)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    results = model(frame)  
    detections = results.pandas().xyxy[0]
    detections = detections[detections["confidence"] > 0.5]  


    print(detections)  

    for _, row in detections.iterrows():
        object_name = row['name']
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        color = (255,0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{object_name} ({row['confidence']:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLOv5 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
