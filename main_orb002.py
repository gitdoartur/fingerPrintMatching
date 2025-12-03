import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


# ----------------------------------------------------------------------
#   Função: preprocess_fingerprint
# ----------------------------------------------------------------------

def preprocess_fingerprint(image_path):

    img = cv2.imread(image_path, 0)

    _, img_bin = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return img_bin


# ----------------------------------------------------------------------
#   Função: match_fingerprints(img1_path, img2_path)
# ----------------------------------------------------------------------

def match_fingerprints(img1_path, img2_path):

    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)

    orb = cv2.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return len(good_matches), match_img


# ----------------------------------------------------------------------
#   Função modificada: process_dataset
# ----------------------------------------------------------------------

def process_dataset(dataset_path, results_folder):

    # Armazena contagens de matches e rótulos verdadeiros
    match_counts = []
    y_true = []

    os.makedirs(results_folder, exist_ok=True)

    # ------------------------------------------------------------
    # Primeiro PASSO: processa todas as pastas e salva match_counts
    # ------------------------------------------------------------

    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):

            image_files = [f for f in os.listdir(folder_path)
                           if f.endswith(('.tif', '.png', '.jpg'))]

            if len(image_files) != 2:
                print(
                    f"Skipping {folder}, expected 2 images but found {len(image_files)}")
                continue

            img1_path = os.path.join(folder_path, image_files[0])
            img2_path = os.path.join(folder_path, image_files[1])

            match_count, match_img = match_fingerprints(img1_path, img2_path)

            # Salva número de matches para testes posteriores
            match_counts.append(match_count)

            # Ground truth
            actual_match = 1 if "same" in folder.lower() else 0
            y_true.append(actual_match)

            # Salva imagem de comparação
            result_img_path = os.path.join(
                results_folder, f"{folder}_matches.png")
            if match_img is not None:
                cv2.imwrite(result_img_path, match_img)

    # ------------------------------------------------------------
    # Segundo PASSO: testar vários thresholds
    # ------------------------------------------------------------
    thresholds = range(0, max(match_counts) + 5, 2)  # ex.: 0,2,4,...
    accuracies = []

    for t in thresholds:
        y_pred = [1 if m > t else 0 for m in match_counts]
        acc = accuracy_score(y_true, y_pred)
        accuracies.append(acc)

    # ------------------------------------------------------------
    # Plot: Acurácia × Threshold
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accuracies, marker='o')
    plt.xlabel("Threshold (número de good matches)")
    plt.ylabel("Acurácia")
    plt.title("Acurácia em função do threshold")
    plt.grid(True)
    plt.show()

    # ------------------------------------------------------------
    # Confusion matrix para o threshold padrão (opcional)
    # ------------------------------------------------------------
    best_t = thresholds[np.argmax(accuracies)]
    y_pred_best = [1 if m > best_t else 0 for m in match_counts]

    print(f"\nMelhor threshold encontrado: {best_t}")
    print(f"Acurácia máxima: {max(accuracies):.4f}")

    cm = confusion_matrix(y_true, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Different", "Same"])
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de Confusão (Threshold ótimo = {best_t})")
    plt.show()


# Execução
dataset_path = r".\Data_check"
results_folder = r".\orb_bf_"
process_dataset(dataset_path, results_folder)
