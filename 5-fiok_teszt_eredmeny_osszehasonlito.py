"""
A script összehasonlítja az eredeti loader élkereső és a YOLO által visszaadott élkereső tengelyeit.
Az eredeti loaderes "original", a YOLOS-s "yolo" kezdetű oszlopokat kapott.
Az eltérések hisztogrammon lettek ábrázolva.
A +- 15 pixel eltéréssel rendelkező detekciók df_erdekesek dataframe-ben tárolódnak.
Az ebben szereplő képekre rákerül
 - zöld színnel a kép felezőjén a névleges x és y tengely
 - kék színnel a YOLO által kapott és számolt tengelykereszt
 - piros színnel az eredeti élkereső algoritmus által visszaadott tengelykereszt
Az 1226 teszt képből 46 darab kép lett ezáltal kiemelve.
Az első 9 képből 3x3 mozaikot készítek.
"""

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
import glob
import numpy as np
import shutil
import random
from fiok_train_helper_functions import read_txt_file

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
sns.set(style="whitegrid")

CAM_PIXEL_RATIO = 10
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

file_path = "./fiok_teszt/Homlokfalkereses_eredmenyei.txt"

class ImageLineDrawer:
    """
        Függőleges és vízszintes vonalakat rajzol egy képre.
        A vonalak a különböző koordináta rendszerek tengelyeit jelképezik:
        - Zöld szín a névleges tengelyeket a kép közepére.
        - Piros szín a meglévő loader kamerázás eredményét rajzolja ki.
        - Kék szín pedig a YOLO eredményét rajzolja ki.
        """
    def __init__(self, image_folder=".\\fiok_teszt\\test_images"):
        self.image_folder = image_folder

    def load_image(self, filename):
        path = os.path.join(self.image_folder, filename)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Nem található a kép: {path}")
        return image, path

    def draw_x_line(self, image, x, color):
        "Függőleges vonal rajzolása."
        cv2.line(image, (x, 0), (x, image.shape[0]), color, 2)
        return image

    def draw_y_line(self, image, y, color):
        "Vízszintes vonal rajzolása."
        cv2.line(image, (0, y), (image.shape[1], y), color, 2)
        return image

    def draw_center_lines(self, image, color = (0, 255, 0)):
        "Centerline megrajzolása a kép közepére."
        cv2.line(image, (int(image.shape[1]/2), 0), (int(image.shape[1]/2), int(image.shape[0])), color, 2)
        cv2.line(image, (0, int(image.shape[0]/2)), (int(image.shape[1]), int(image.shape[0]/2)), color, 2)
        return image

    def image_processing(self, filename, x, y, x_original, y_original, color = (255, 0, 0)):
        """
        :param filename: feldolgozandó fájl név
        :param x: YOLO függőleges tengely pozíciója
        :param y: YOLO vízszintes tengely pozíciója
        :param x_original: eredeti loader script függőleges tengely pozíciója
        :param y_original: eredeti loader script vízszintes tengely pozíciója
        :param color: szín opcionálisan (zöld, kék, piros)
        :return: semmi, csak lementi a megfelelő almappába a képet a berajzolt tengelyekkel
        """
        #kép betöltése mappából
        img, path = self.load_image(filename)
        #névleges tengelyek berajzolása zöld színnel
        img = self.draw_center_lines(img)
        #YOLO tengelyek berajzolása kék színnel
        img = self.draw_x_line(img, x, color)
        img = self.draw_y_line(img, y, color)
        #Meglévő script által visszaadott tengelyek berajzolása pirossal
        img = self.draw_x_line(img, x_original, color = (0, 0, 255))
        img = self.draw_y_line(img, y_original, color = (0, 0, 255))

        cv2.putText(img,
                    filename,
                    (10, 30),  # X, Y koordináta (bal felső sarokhoz közel)
                    cv2.FONT_HERSHEY_SIMPLEX,  # Betűtípus
                    0.8,  # Betűméret
                    (0, 255, 0),  # Szín (B, G, R) -> zöld
                    1,  # Vonalvastagság
                    cv2.LINE_AA)  # Anti-aliased vonal
        #Kép mentése
        save_path = os.path.join(os.getcwd(), "fiok_teszt", "osszehasonlito_kepek", filename)

        cv2.imwrite(f"{save_path}", img)
        print(f"#INFO: File name: {filename}. Mentés kész.")


tartalom = read_txt_file(file_path)[14:]
data = []

print("Hibás képek:")
for i,line in enumerate(tartalom):
    line_data = {
        "file_name": None,
        "x": None,
        "y": None
    }
    if line == "Kep felolvasva:":
        line_data = {
            "file_name": tartalom[i+1].split("/")[-1],
            "x": int(tartalom[i+4].split(":")[-1].strip()),
            "y": int(tartalom[i+5].split(":")[-1].strip())
        }
        data.append(line_data)
        #print(f"Sor: {i}: {line_data}")
    if line.startswith("Kep felolvasas sikertelen!"):
        line_data = {
            "file_name": line.split("/")[-1].split(")")[0],
            "x": None,
            "y": None
        }
        print(f"{i}: {line_data}")
        data.append(line_data)

print("\nOriginal dataframe:")
df_original = pd.DataFrame(data, columns=["file_name", "x", "y"])
df_original.columns = ["file_name", "original_x_offset", "original_y_offset"]
df_original["original_x_pixel"] = (df_original["original_x_offset"] * 10 ) + 1280 / 2
df_original["original_y_pixel"] = (df_original["original_y_offset"] * 10 ) + 720 / 2
print(df_original.head())

print("#INFO: results.csv beolvasása")
df_eredmeny = pd.read_csv("./fiok_teszt/results.csv", delimiter=',')
df_eredmeny.columns =  ["file_name", "yolo_x_pixel", "yolo_y_pixel"]
df_eredmeny["yolo_x_offset"] = (df_eredmeny["yolo_x_pixel"] - 640) / 10
df_eredmeny["yolo_y_offset"] = (df_eredmeny["yolo_y_pixel"] - 360) / 10
print("\nYOLO eredmény df")
print(df_eredmeny.head())

df = df_eredmeny.merge(df_original, on='file_name', how='outer')
#df = df[["file_name", "yolo_x_offset",  "yolo_y_offset",  "original_x_offset",  "original_y_offset"]]
df["x_diff"] = df["yolo_x_offset"] - df["original_x_offset"]
df["y_diff"] = df["yolo_y_offset"] - df["original_y_offset"]

#kerekítések a vonalak rajzolásához:
df["yolo_x_pixel"] = df["yolo_x_pixel"].round().astype("Int64")
df["yolo_y_pixel"] = df["yolo_y_pixel"].round().astype("Int64")
df["original_x_pixel"] = df["original_x_pixel"].round().astype("Int64")
df["original_y_pixel"] = df["original_y_pixel"].round().astype("Int64")

print("\nKombinált df")
print(df.head())
print(df.describe())

df["x_diff"].hist(bins=40, edgecolor="black")
# Megjelenítés
plt.xlabel("Érték")
plt.ylabel("Gyakoriság")
plt.title("Hisztogram az 'x_diff' oszlopról")
#plt.show()

df["y_diff"].hist(bins=40, edgecolor="black")
# Megjelenítés
plt.xlabel("Érték")
plt.ylabel("Gyakoriság")
plt.title("Hisztogram az 'y_diff' oszlopról")
#plt.show()

df_erdekesek = df[(df["x_diff"] > 15) | (df["x_diff"] < -15) | (df["y_diff"] > 15) | (df["y_diff"] < -15)]
print(f"#INFO: Érdekes df: {df_erdekesek.shape}")

folder_path = os.path.join(os.getcwd(), "fiok_teszt", "osszehasonlito_kepek")
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
os.makedirs(folder_path)
print(f"#INFO: Mappa létrehozva: {folder_path}")
print("\nÉrdekes df adatok:")
print(df_erdekesek.describe())
print(df_erdekesek)

image_drawer = ImageLineDrawer()
for i in range(df_erdekesek.shape[0]):
    image_drawer.image_processing(
        filename = df_erdekesek.iloc[i, 0],
        x = df_erdekesek.iloc[i, 1],
        y = df_erdekesek.iloc[i, 2],
        x_original = df_erdekesek.iloc[i, 7],
        y_original= df_erdekesek.iloc[i, 8]
    )

#Mozaikhoz a képek

images_for_masaic = os.path.join(folder_path, "*.jpg")
image_paths = sorted(glob.glob(f"{images_for_masaic}"))[:9]  # első 9 kép
print(image_paths)
#image_paths = random.shuffle(glob.glob(f"{images_for_masaic}"))[:9]  # első 9 kép
images = [cv2.imread(p) for p in image_paths]

# 3×3-as mozaik létrehozása
row1 = np.hstack(images[0:3])
row2 = np.hstack(images[3:6])
row3 = np.hstack(images[6:9])
mosaic = np.vstack([row1, row2, row3])

# Mozaik mentése
cv2.imwrite("mosaic.jpg", mosaic)



"""
# Q–Q plot
fig = sm.qqplot(df["x_diff"], line='s')  # 's' = standardizált vonal
plt.title("Q–Q plot az 'x_diff' oszlopról", fontsize=14)
plt.xlabel("Elméleti kvantilisek")
plt.ylabel("Mintabeli kvantilisek")
plt.show()
fig = sm.qqplot(df["y_diff"], line='s')  # 's' = standardizált vonal
plt.title("Q–Q plot az 'y_diff' oszlopról", fontsize=14)
plt.xlabel("Elméleti kvantilisek")
plt.ylabel("Mintabeli kvantilisek")
plt.show()

"""