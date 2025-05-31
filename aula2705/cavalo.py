import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

video_path = 'videoYOLO.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('saida_cavalos.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        # Classe 17 = Horse
        if cls == 17 and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Horse {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()