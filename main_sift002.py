import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


# ------------------------------------------------------------
# Pré-processamento da impressão digital
# ------------------------------------------------------------

def preprocess_fingerprint(image_path):
    img = cv2.imread(image_path, 0)
    _, img_bin = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin


# ------------------------------------------------------------
# Comparação entre duas digitais usando SIFT + FLANN
# ------------------------------------------------------------

def match_fingerprints(img1_path, img2_path):

    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)

    sift = cv2.SIFT_create(nfeatures=1000)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, None

    index_params = dict(algorithm=1, trees=5)  # KD-tree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return len(good_matches), match_img


# ------------------------------------------------------------
# Avaliação do dataset + análise de thresholds
# ------------------------------------------------------------

def process_dataset(dataset_path, results_folder):

    os.makedirs(results_folder, exist_ok=True)

    match_counts = []
    y_true = []

    # ------------------------------------------------------------
    # 1º PASSO — processar pastas e armazenar match_counts
    # ------------------------------------------------------------
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)

        if os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path)
                           if f.endswith(('.tif', '.png', '.jpg'))]

            if len(image_files) != 2:
                print(
                    f"Skipping {folder}, found {len(image_files)} images instead of 2.")
                continue

            img1_path = os.path.join(folder_path, image_files[0])
            img2_path = os.path.join(folder_path, image_files[1])

            match_count, match_img = match_fingerprints(img1_path, img2_path)

            match_counts.append(match_count)

            actual_match = 1 if "same" in folder.lower() else 0
            y_true.append(actual_match)

            # Salva imagem com matches
            output_path = os.path.join(results_folder, f"{folder}_matches.png")
            if match_img is not None:
                cv2.imwrite(output_path, match_img)

    # ------------------------------------------------------------
    # 2º PASSO — Avaliar vários thresholds
    # ------------------------------------------------------------
    thresholds = range(0, max(match_counts) + 5, 2)
    accuracies = []

    for t in thresholds:
        y_pred = [1 if m > t else 0 for m in match_counts]
        acc = accuracy_score(y_true, y_pred)
        accuracies.append(acc)

    # ------------------------------------------------------------
    # Plotar acurácia x threshold
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accuracies, marker='o')
    plt.xlabel("Threshold (número de good matches)")
    plt.ylabel("Acurácia")
    plt.title("Acurácia em função do threshold – SIFT + FLANN")
    plt.grid(True)
    plt.show()

    # ------------------------------------------------------------
    # 3º PASSO — Melhor threshold
    # ------------------------------------------------------------
    best_t = thresholds[np.argmax(accuracies)]
    best_acc = max(accuracies)

    print(f"\nMelhor threshold: {best_t}")
    print(f"Acurácia máxima: {best_acc:.4f}")

    # Predições com o threshold ótimo
    y_pred_best = [1 if m > best_t else 0 for m in match_counts]

    # ------------------------------------------------------------
    # 4º PASSO — Matriz de confusão com o threshold ótimo
    # ------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Different (0)", "Same (1)"])
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix – SIFT + FLANN (Threshold = {best_t})")
    plt.show()


# ------------------------------------------------------------
# Execução
# ------------------------------------------------------------
dataset_path = r".\Data_check"
results_folder = r".\sift_flann_"

process_dataset(dataset_path, results_folder)
