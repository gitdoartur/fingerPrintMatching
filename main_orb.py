
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ----------------------------------------------------------------------
#   Função: preprocess_fingerprint
# ----------------------------------------------------------------------
#   Objetivo: Pré-processar impressões digitais.

def preprocess_fingerprint(image_path):

    # Lê a imagem em escala de cinza (0).
    img = cv2.imread(image_path, 0)

    # Aplica limiarização binária inversa com Otsu.
    _, img_bin = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Retorna a imagem binarizada.
    return img_bin


# ----------------------------------------------------------------------
#   Função: match_fingerprints(img1_path, img2_path)
# ----------------------------------------------------------------------
#   Objetivo: Encontrar número de correspondências (matches) entre duas digitais.

def match_fingerprints(img1_path, img2_path):

    # Pré-processa as duas imagens.
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)

    # Inicializa o detector ORB para extrair até 1000 pontos de interesse.
    orb = cv2.ORB_create(nfeatures=1000)

    # Encontra keypoints e descritores
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, None  # Retorna 0 matches caso nenhum descritor tenha sido achado

    # Usa o comparador por força-bruta com a distância de Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # KNN Match
    matches = bf.knnMatch(des1, des2, k=2)

    # Aplica ao teste da razão de Lowe (Lowe's ratio test): mantém apenas os bons matches
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Desenha os matches (bons) encontrados
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches,
                                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), match_img
