import cv2
import numpy as np

imagem = cv2.imread("cartao.jpg")

camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")
undistorted = cv2.undistort(imagem, camera_matrix, dist_coeffs)

pontos = []

def clique(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pontos) < 4:
        pontos.append((x, y))
        print(f"🖱️ Ponto {len(pontos)}: ({x}, {y})")
        cv2.circle(undistorted, (x, y), 5, (0, 0, 255), -1)
        if len(pontos) in [2, 4]:
            cv2.line(undistorted, pontos[-2], pontos[-1], (255, 0, 0), 2)
        cv2.imshow("Imagem", undistorted)

print("Clique nos dois extremos do cartão de crédito")
print("Depois clique nos dois extremos do objeto")
print("Feche a janela para ver o resultado")

cv2.imshow("Imagem", undistorted)
cv2.setMouseCallback("Imagem", clique)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(pontos) != 4:
    print("⚠️ Foram detectados apenas", len(pontos), "pontos. Precisamos de 4.")
else:
    ref_px = np.linalg.norm(np.array(pontos[0]) - np.array(pontos[1]))
    obj_px = np.linalg.norm(np.array(pontos[2]) - np.array(pontos[3]))
    ref_mm = 85.60
    escala = ref_mm / ref_px
    medida_mm = obj_px * escala

    print(f"\n📏 Referência: {ref_px:.2f} px = {ref_mm} mm")
    print(f"📦 Objeto: {obj_px:.2f} px")
    print(f"✅ Tamanho estimado do objeto: {medida_mm:.2f} mm")
