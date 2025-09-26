"""
1. Betanító script.
Ez az először lefuttatandó fájl:
- elkészíti a train és teszt adathalmazt,
- megtörténik a betanítás
- lementi a modellt
Ne felejtsd el az "mlflow ui" parancsot beírni a konzolba!

"""

import cv2
import torch
torch.classes.__path__ = []

import os
from ultralytics import YOLO
import shutil
import time
import mlflow
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from fiok_train_helper_functions import load_yaml_config
from fiok_train_helper_functions import adjust_brightness_contrast, histogram_equalization, clahe, denoise_image
from fiok_train_helper_functions import create_mosaic
from fiok_train_helper_functions import train_data_flip, setup_yolov11_dataset

from ultralytics import settings
# Return a specific setting
value = settings["runs_dir"]

print("#INFO: Importok OK")

config_path = "fiok_train/config.yaml"  # Cseréld ki a megfelelő fájlútra
config = load_yaml_config(config_path)

if config:
    print("Konfiguráció tartalma:")
    print(config)

train_dir_del = os.path.join(config["input_dir"], "train")
val_dir_del = os.path.join(config["input_dir"], "val")
try:
    shutil.rmtree(train_dir_del)
    shutil.rmtree(val_dir_del)
    print("#INFO: Meglévő Train és Val könyvtárak törölve!")
except:
    print("#INFO: Nem volt train és val könyvtár!")

# Kép betöltése
image = cv2.imread('fiok_train/Frame_20250203_100349_212_LoaderFS1009FK_StorageHangerFaceplateStateDetect_Left.jpg')

# Fényviszonyok javítása
bright_contrast_image = adjust_brightness_contrast(image, alpha=1.5, beta=30)
histogram_image = histogram_equalization(image)
clahe_image = clahe(image)

# Zajcsökkentés alkalmazása minden módosított képre
bright_contrast_image_denoised = denoise_image(bright_contrast_image)
histogram_image_denoised = denoise_image(histogram_image)
clahe_image_denoised = denoise_image(clahe_image)

# Képek megjelenítése
#cv2.imshow("Original Image", image)
#cv2.imshow("Brightness and Contrast Adjusted", bright_contrast_image)
#cv2.imshow("Histogram Equalized", histogram_image)
#cv2.imshow("CLAHE Applied", clahe_image)

#cv2.imshow("Denoised Brightness and Contrast Adjusted", bright_contrast_image_denoised)
#cv2.imshow("Denoised Histogram Equalized", histogram_image_denoised)
#cv2.imshow("Denoised CLAHE Applied", clahe_image_denoised)

# Kép mentése, ha szükséges
cv2.imwrite("fiok_train/_adjusted_brightness_contrast.jpg", bright_contrast_image)
cv2.imwrite("fiok_train/_histogram_equalized.jpg", histogram_image)
cv2.imwrite("fiok_train/_clahe_image.jpg", clahe_image)
cv2.imwrite("fiok_train/_denoised_brightness_contrast.jpg", bright_contrast_image_denoised)
cv2.imwrite("fiok_train/_denoised_histogram_equalized.jpg", histogram_image_denoised)
cv2.imwrite("fiok_train/_denoised_clahe_image.jpg", clahe_image_denoised)
print("#INFO: Összehasonlító képek mentése kész")

#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Képek és feliratok listája
images = [
    bright_contrast_image,
    histogram_image,
    clahe_image,
    bright_contrast_image_denoised,
    histogram_image_denoised,
    clahe_image_denoised
]

titles = [
    "Brightness & Contrast Adjusted",
    "Histogram Equalized",
    "CLAHE Applied",
    "Denoised Brightness & Contrast",
    "Denoised Histogram Equalized",
    "Denoised CLAHE"
]

# Mozaik létrehozása (2 sor, 3 oszlop)
mosaic_image = create_mosaic(images, titles, rows=2, cols=3)

# Mozaik kép megjelenítése
cv2.imshow("Mosaic Image", cv2.resize(mosaic_image, (1280, 720)))

# Kép mentése
cv2.imwrite("fiok_train/mosaic_image.jpg", mosaic_image)
print("#INFO: Mozaik kép mentése kész!")

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.resize(mosaic_image, (1920, 1080))
# Matplotlib segítségével a mozaik képek vizuális megjelenítése
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(mosaic_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Mosaic of Image Processing Methods")
plt.show()

setup_yolov11_dataset(config["input_dir"], config["output_dir"], split_ratio=0.8)

if config['flip'] == "yes":
    train_data_flip()

stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
run_dirs = os.path.join('fiok_train', f"train_{stamp}")
print("#INFO: Betanítás kezdődik:")
model = YOLO("fiok_train/yolo11n.pt")

EPOCHS=30
IMGSZ=640
BATCH=4

mlflow.set_experiment("loader-fiok-yolo-train")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
with mlflow.start_run():
    train_results = model.train(
        data="dataset_custom.yaml",  # path to dataset YAML
        epochs=EPOCHS,  # number of training epochs
        imgsz=IMGSZ,  # training image size
        device='cpu',  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=BATCH,
        workers=0,
        seed=42,
        project=run_dirs
        # patience = 8,
        # name = "11s_clahe"
    )
    """
    # Paraméterek logolása
    mlflow.log_params({
        "model": "yolo11n",
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "img_size": IMGSZ,
        "device": "cpu",
        "lr": 0.001,
        "seed": 42,
        "workers": 0
    })
    mlflow.log_metrics({
        "final_mAP50": train_results.results_dict['metrics/mAP50(B)'],
        "final_mAP50-95": train_results.results_dict['metrics/mAP50-95(B)'],
        "final_precision": train_results.results_dict['metrics/precision(B)'],
        "final_recall": train_results.results_dict['metrics/recall(B)']
    })
    """
    # Modell mentése
    mlflow.pytorch.log_model(model.model, f"train_{stamp}")
