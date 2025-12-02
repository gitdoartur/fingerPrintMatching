
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


# ----------------------------------------------------------------------
#   Função: process_dataset(dataset_path, results_folder)
# ----------------------------------------------------------------------
#   Comparar pares de imagens em várias pastas e avaliar desempenho da técnica ORB + BF + Lowe.

def process_dataset(dataset_path, results_folder):
    # Inicialização
    threshold = 20  # Deve ser ajustado baseado em testes
    y_true = []  # True labels (1 for same, 0 for different)
    y_pred = []  # Predicted labels

    # Cria uma pasta de resultados caso não exista
    os.makedirs(results_folder, exist_ok=True)

    # Loop sobre todos os subdiretórios
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):  # checa se é um diretório válido
            image_files = [f for f in os.listdir(
                folder_path) if f.endswith(('.tif', '.png', '.jpg'))]
            if len(image_files) != 2:
                print(
                    f"Skipping {folder}, expected 2 images but found {len(image_files)}")
                continue  # Prossegue caso a pasta não tenha exatamente 2 imagens
            img1_path = os.path.join(folder_path, image_files[0])
            img2_path = os.path.join(folder_path, image_files[1])
            match_count, match_img = match_fingerprints(img1_path, img2_path)

            # Determine the ground truth (expected label)
            actual_match = 1 if "same" in folder.lower() else 0  # 1 for same, 0 for different
            y_true.append(actual_match)

            # Decision based on good matches count
            predicted_match = 1 if match_count > threshold else 0
            y_pred.append(predicted_match)
            result = "orb_bf_matched" if predicted_match == 1 else "orb_bf_unmatched"
            print(f"{folder}: {result.upper()} ({match_count} good matches)")

            # Save match image in the results folder
            if match_img is not None:
                match_img_filename = f"{folder}_{result}.png"
                match_img_path = os.path.join(
                    results_folder, match_img_filename)
                cv2.imwrite(match_img_path, match_img)
                print(f"Saved match image at: {match_img_path}")

    # Calcula e apresetna a matriz de confusão
    labels = ["Different (0)", "Same (1)"]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix orb_bf")
    plt.show()


# Examplo de uso
dataset_path = r".\Data_check"
results_folder = r".\orb_bf_"
process_dataset(dataset_path, results_folder)
