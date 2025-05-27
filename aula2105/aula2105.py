import cv2
import numpy as np
import time
import os

# Caminhos dos arquivos
# Os arquivos estão na mesma pasta
model_file = 'pose_iter_160000.caffemodel'
proto_file = 'pose_deploy_linevec_faster_4_stages.prototxt'

# Verificação se o arquivo está ou não na pasta
if not os.path.exists(model_file) or not os.path.exists(proto_file):
    raise FileNotFoundError("Modelo ou prototxt não encontrado.")

# Carrega a rede
network = cv2.dnn.readNetFromCaffe(proto_file, model_file)
print("Rede carregada com sucesso.")

# Conexões entre pontos
conexoes = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14],
            [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

NUM_PONTOS = 15
THRESHOLD = 0.1

# Funções de detecção
def detectar_perna(pontos):
    msgs = []
    if pontos[8] and pontos[9] and pontos[10]:
        if pontos[9][1] < pontos[8][1] or pontos[10][1] < pontos[8][1]:
            msgs.append("Perna direita levantada")
    if pontos[11] and pontos[12] and pontos[13]:
        if pontos[12][1] < pontos[11][1] or pontos[13][1] < pontos[11][1]:
            msgs.append("Perna esquerda levantada")
    return msgs

def detectar_braco(pontos):
    msgs = []
    if pontos[2] and pontos[3] and pontos[4]:
        if pontos[3][1] < pontos[2][1] or pontos[4][1] < pontos[2][1]:
            msgs.append("Braco direito levantado")
    if pontos[5] and pontos[6] and pontos[7]:
        if pontos[6][1] < pontos[5][1] or pontos[7][1] < pontos[5][1]:
            msgs.append("Braco esquerdo levantado")
    return msgs

# Inicialização da câmera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Tamanho de entrada da rede
input_size = (256, 256)

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    resized = cv2.resize(frame, input_size)

    blob = cv2.dnn.blobFromImage(resized, 1.0 / 255, input_size, (0, 0, 0), swapRB=False, crop=False)
    network.setInput(blob)
    output = network.forward()

    H_out, W_out = output.shape[2], output.shape[3]
    escala_x = w / W_out
    escala_y = h / H_out

    pontos = []
    for i in range(NUM_PONTOS):
        mapa = output[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(mapa)
        x = int(point[0] * escala_x)
        y = int(point[1] * escala_y)
        if conf > THRESHOLD:
            pontos.append((x, y))
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        else:
            pontos.append(None)

    for a, b in conexoes:
        if pontos[a] and pontos[b]:
            cv2.line(frame, pontos[a], pontos[b], (255, 0, 0), 2)

    # Detecta ações
    msgs = detectar_perna(pontos) + detectar_braco(pontos)
    for i, m in enumerate(msgs):
        cv2.putText(frame, m, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Mostra frame
    cv2.imshow("Deteccao", frame)

    # Controlar o FPS
    key = cv2.waitKey(1)
    elapsed = time.time() - start
    if elapsed < 0.06:
        time.sleep(0.06 - elapsed)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
