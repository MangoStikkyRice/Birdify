from ultralytics import YOLO
import cv2
import os

# Paths
video_path = "videos/video1.mp4"  
output_video_path = "output_video_with_boxes.mp4"  

model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(video_path)


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    results = model(frame) 

    detections = results[0].boxes.xyxy
    confidences = results[0].boxes.conf 
    classes = results[0].boxes.cls
    

    for i in range(len(detections)):
        x1, y1, x2, y2 = map(int, detections[i])
        conf = confidences[i]
        cls = int(classes[i])
        label = f"{model.names[cls]} {conf:.2f}"
        

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    out.write(frame)
    

    cv2.imshow("Bird Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved as {output_video_path}")
