import cv2
import matplotlib.pyplot as plt

enderecoImagem = 'D://Guilherme//Downloads//imagem.jpg'
imagem = cv2.imread(enderecoImagem)

#apresentar erro caso nao carregue imagem
if imagem is None:
    print("Erro: imagem não encontrada. Verifique o caminho.")
else:
    #mostrar a imagem com keypoints
    def mostrarImagemComKeypoints(imagem, keypoints, titulo):
        imagemComPontos = imagem.copy()
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(imagemComPontos, (x, y), 1, (0, 0, 0), -1)
        imagemRgb = cv2.cvtColor(imagemComPontos, cv2.COLOR_BGR2RGB)
        plt.imshow(imagemRgb)
        plt.title(titulo)
        plt.show()

    #Redimensionar a imagem
    novaLargura = 300
    novaAltura = 200
    imagemRedimensionada = cv2.resize(imagem, (novaLargura, novaAltura))

    #Rotacionar a imagem
    alturaImagem, larguraImagem = imagem.shape[:2]
    centroImagem = (larguraImagem // 2, alturaImagem // 2)
    anguloRotacao = 45
    escala = 1.0
    matrizRotacao = cv2.getRotationMatrix2D(centroImagem, anguloRotacao, escala)
    imagemRotacionada = cv2.warpAffine(imagem, matrizRotacao, (larguraImagem, alturaImagem))

    #escala de cinza
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagemCinzaRedimensionada = cv2.cvtColor(imagemRedimensionada, cv2.COLOR_BGR2GRAY)
    imagemCinzaRotacionada = cv2.cvtColor(imagemRotacionada, cv2.COLOR_BGR2GRAY)

    #Criar o detector SIFT e detectar os pontos-chave
    sift = cv2.SIFT_create()

    #Detecta keypoints
    keypointsOriginal = sift.detect(imagemCinza, None)
    keypointsRedimensionada = sift.detect(imagemCinzaRedimensionada, None)
    keypointsRotacionada = sift.detect(imagemCinzaRotacionada, None)

    #Mostrar as 3 imagens com keypoints
    mostrarImagemComKeypoints(imagem, keypointsOriginal, "Imagem Original")
    mostrarImagemComKeypoints(imagemRedimensionada, keypointsRedimensionada,"Imagem Redimensionada")
    mostrarImagemComKeypoints(imagemRotacionada, keypointsRotacionada, "Imagem Rotacionada")

    #Quantidade de keypoints detectados
    print(f"Quantidade de keypoints na imagem original: {len(keypointsOriginal)}")
    print(f"Quantidade de keypoints na imagem redimensionada: {len(keypointsRedimensionada)}")
    print(f"Quantidade de keypoints na imagem rotacionada: {len(keypointsRotacionada)}")

    #Localização dos keypoints
    print("\nLocalizações dos keypoints na imagem original:")
    for kp in keypointsOriginal[:5]:
        print(f"({kp.pt[0]}, {kp.pt[1]})")

    print("\nLocalizações dos keypoints na imagem redimensionada:")
    for kp in keypointsRedimensionada[:5]:
        print(f"({kp.pt[0]}, {kp.pt[1]})")

    print("\nLocalizações dos keypoints na imagem rotacionada:")
    for kp in keypointsRotacionada[:5]:
        print(f"({kp.pt[0]}, {kp.pt[1]})")