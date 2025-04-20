import cv2
import torch

model = torch.hub.load('ultralytics/yolov5','yolov5s')

cap = cv2.VideoCapture(r"C:\Users\prasa\OneDrive\Desktop\YOLO\YOLOv8\YOLO\video_sample1.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print('error:could not open video file')
    exit()

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print('End of video or error reading frame')
        break

    frame_count += 1
    print(f'processing frame {frame_count}')

    result = model(frame)
    detections = result.pandas().xyxy[0]

    for _, row in detections.iterrows():
        object_name = row['name']

        if object_name in ('car', 'bicycle'):
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            color = (0,255,0) if object_name == 'car' else (255,0,0) if object_name == 'bicycle' else (0,0,255)

            cv2.rectangle(frame, (x1,y1),(x2,y2),color,2)

            label = f"{object_name} ({row['confidence']:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow('yolo object detection',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()