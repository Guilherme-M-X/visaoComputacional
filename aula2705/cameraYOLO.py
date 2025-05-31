import cv2
from ultralytics import YOLO

# Carregue o modelo YOLOv8 pré-treinado (coco)
model = YOLO('yolov8n.pt')

# Inicie a webcam (0 = webcam padrão)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realize a detecção
    results = model(frame)

    # Percorra as detecções
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        # Classe 0 é 'person' no COCO
        if cls == 0 and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('Webcam - Pessoa', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()