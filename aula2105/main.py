import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        h, w, _ = frame.shape

        # Pega coordenadas
        head = lm[mp_pose.PoseLandmark.NOSE]
        shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Desenha os pontos (tudo verde)
        for point in [head, shoulder, wrist]:
            cx, cy = int(point.x * w), int(point.y * h)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

        # Verifica se o punho tá acima do ombro
        if wrist.y < shoulder.y:
            cv2.putText(frame, "Levantado", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "Abaixado", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
