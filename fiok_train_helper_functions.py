import yaml
import cv2
import os
import glob
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import random
import shutil

def read_txt_file(file_path):
    """
    Beolvas egy txt fájlt és visszaadja a tartalmát.

    Args:
        file_path (str): A txt fájl elérési útja

    Returns:
        str: A fájl tartalma, vagy None ha hiba történt
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # Sorok végéről eltávolítjuk a \n karaktereket
            lines = [line.strip() for line in lines]
            return lines
    except FileNotFoundError:
        print(f"Hiba: A fájl nem található: {file_path}")
        return None
    except PermissionError:
        print(f"Hiba: Nincs jogosultság a fájl olvasásához: {file_path}")
        return None
    except UnicodeDecodeError:
        print(f"Hiba: Nem sikerült dekódolni a fájlt UTF-8 kódolással: {file_path}")
        # Próbálkozás más kódolással
        try:
            with open(file_path, 'r', encoding='cp1252') as file:
                content = file.read()
                print("Sikeresen beolvasva cp1252 kódolással.")
                return content
        except:
            print("Nem sikerült más kódolással sem beolvasni.")
            return None
    except Exception as e:
        print(f"Váratlan hiba történt: {e}")
        return None


def setup_yolov11_dataset(input_dir, output_dir, split_ratio=0.8):
    """
    Létrehozza a YOLOv11 betanításhoz szükséges könyvtárszerkezetet és szétosztja az adatokat.

    :param input_dir: Az a könyvtár, amely tartalmazza a .jpg és .txt fájlokat.
    :param output_dir: Az a könyvtár, ahol a kimeneti struktúra létrejön.
    :param split_ratio: A tréning és validációs adathalmaz aránya (pl. 0.8 = 80% tréning).
    """
    # Ellenőrizze, hogy az input könyvtár létezik-e
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"A megadott bemeneti könyvtár nem létezik: {input_dir}")

    # YOLOv8 könyvtárszerkezet létrehozása
    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    val_images_dir = os.path.join(output_dir, "val", "images")
    val_labels_dir = os.path.join(output_dir, "val", "labels")

    for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(directory, exist_ok=True)

    # Az összes .jpg és .txt fájl beolvasása
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    label_files = glob.glob(os.path.join(input_dir, "*.txt"))

    # Az azonos nevű fájlok párosítása
    base_names = set(os.path.splitext(os.path.basename(f))[0] for f in image_files)
    paired_files = [(os.path.join(input_dir, f + ".jpg"), os.path.join(input_dir, f + ".txt"))
                    for f in base_names if os.path.join(input_dir, f + ".txt") in label_files]

    # A fájlok véletlenszerű sorrendbe állítása
    random.shuffle(paired_files)

    # Az adatok szétosztása tréning és validációs halmazokra
    split_index = int(len(paired_files) * split_ratio)
    train_files = paired_files[:split_index]
    val_files = paired_files[split_index:]

    # Fájlok átmásolása a megfelelő könyvtárakba
    for image_path, label_path in train_files:
        shutil.copy(image_path, train_images_dir)
        shutil.copy(label_path, train_labels_dir)

    for image_path, label_path in val_files:
        shutil.copy(image_path, val_images_dir)
        shutil.copy(label_path, val_labels_dir)

    print(f"#INFO: Adatok sikeresen szétosztva és átmásolva a YOLOv11 könyvtárszerkezetbe.")
    print(f"#INFO: Tréning adatok: {len(train_files)} fájl, Validációs adatok: {len(val_files)} fájl.")

    files = glob.glob(os.path.join(train_images_dir, "*.jpg"))
    for img in files:
        raw_image = cv2.imread(img)
        clahe_image = clahe(raw_image)
        cv2.imwrite(img, clahe_image)

    files = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    for img in files:
        raw_image = cv2.imread(img)
        clahe_image = clahe(raw_image)
        cv2.imwrite(img, clahe_image)

    print("#INFO: Képek előfeldolgozása CLAHE algoritmussal kész!")

def load_yaml_config(file_path):
    """
    Beolvassa a YAML konfigurációs fájlt.

    :param file_path: Az elérési út a YAML fájlhoz.
    :return: A YAML fájl tartalma Python dictionary-ként.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)  # A safe_load biztonságosabb, mint a load
            return config
    except FileNotFoundError:
        print(f"Hiba: A fájl nem található: {file_path}")
    except yaml.YAMLError as e:
        print(f"Hiba történt a YAML beolvasásakor: {e}")

def adjust_brightness_contrast(image, alpha=1.2, beta=20):
    """
    Alkalmaz kontraszt és fényerő módosítást.
    alpha: A kontraszt (1.0 nem változtat semmit, >1.0 növeli, <1.0 csökkenti).
    beta: A fényerő (0 nem változtat semmit, pozitív érték növeli, negatív csökkenti).
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def histogram_equalization(image):
    """
    Histogramaegyenlítés szürkeárnyalatú képeken.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

def clahe(image):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) alkalmazása.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    return cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

def denoise_image(image):
    """
    Zajcsökkentés a képen bilaterális szűréssel.
    """
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def create_mosaic(images, titles, rows, cols):
    """
    Képek és feliratok alapján mozaik létrehozása.
    images: A képek listája.
    titles: A képekhez tartozó feliratok listája.
    rows: A mozaik sorainak száma.
    cols: A mozaik oszlopainak száma.
    """
    # A képek átméretezése, hogy illeszkedjenek a mozaikba
    max_height = max([img.shape[0] for img in images])
    max_width = max([img.shape[1] for img in images])

    # Új, üres mozaik készítése
    mosaic = np.zeros((max_height * rows, max_width * cols, 3), dtype=np.uint8)

    # A képek elhelyezése a mozaikban
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        y_offset = row * max_height
        x_offset = col * max_width
        mosaic[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

        # Felirat hozzáadása a kép alá
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = titles[idx]
        position = (x_offset + 10, y_offset + img.shape[0] - 10)
        cv2.putText(mosaic, text, position, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return mosaic

def train_data_flip():
    # 📂 Könyvtár megadása, ahol a képek és annotációk vannak
    input_dir_images = os.path.join("fiok_train/train", "images")
    input_dir_labels = os.path.join("fiok_train/train", "labels")

    # 🔍 Összegyűjtjük az összes kép fájlt (jpg, png, stb.)
    image_files = glob.glob(os.path.join(input_dir_images, "*.jpg")) + glob.glob(os.path.join(input_dir_images, "*.png"))

    # 🔄 Minden fájlra végrehajtjuk a tükrözést
    for image_path in image_files:
        # 📌 Alapfájlnevek előkészítése
        base_name = os.path.splitext(os.path.basename(image_path))[0]  # Pl.: "image1"
        annotation_path = os.path.join(input_dir_labels, f"{base_name}.txt")  # YOLO annotáció fájl

        # 📌 Kép beolvasása OpenCV-vel
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hiba a kép beolvasásakor: {image_path}")
            continue

        # 📌 Kép vízszintes tükrözése (horizontal flip)
        flipped_image = cv2.flip(image, 1)

        # 📌 Új fájlnevek generálása (_flip kiegészítéssel)
        flipped_image_path = os.path.join(input_dir_images, f"{base_name}_flip.jpg")
        flipped_annotation_path = os.path.join(input_dir_labels, f"{base_name}_flip.txt")

        # 💾 Kép mentése új fájlként
        cv2.imwrite(flipped_image_path, flipped_image)

        # 🔄 Annotáció módosítása, ha létezik a TXT fájl
        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                lines = f.readlines()

            flipped_annotations = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # Hibás sorokat kihagyjuk

                class_id, x_center, y_center, width, height = map(float, parts)

                # 📌 X koordináta tükrözése
                new_x_center = 1.0 - x_center  # X flip: 1 - x_center

                # Új annotáció formázása
                flipped_annotations.append(
                    f"{int(class_id)} {new_x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # 💾 Új annotációs fájl mentése
            with open(flipped_annotation_path, "w") as f:
                f.writelines(flipped_annotations)

    #print(f"Mentve: {flipped_image_path}, {flipped_annotation_path}")
    print("#INFO: Tükrözött képek és módosított txt fájlok mentése kész")

def draw_boxes(image, results):
    image_copy = image.copy()
    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box koordináták
            if y1 > 250:
                continue
            conf = box.conf[0].item()  # Konfidencia érték
            cls = int(box.cls[0])  # Osztály index
            label = result.names[cls]  # Osztály neve

            # Téglalap rajzolása
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Felirat szövege
            text = f"{label} {conf:.2f}"

            # Szöveg mérete és háttér
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image_copy, (x1, y1 - h + 25), (x1 + w, y1 + 25), (0, 255, 0), -1)
            cv2.putText(image_copy, text, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return image_copy

def find_latest_csv_by_timestamp(directory=None):
    """
    Megkeresi az időben legutolsó CSV fájlt a time.time() timestamp alapján

    Args:
        directory: mappa útvonal (None esetén jelenlegi mappa)

    Returns:
        tuple: (fájl_útvonal, timestamp) vagy (None, None) ha nincs találat
    """
    if directory is None:
        directory = os.getcwd()

    print(f"CSV fájlok keresése mappában: {directory}")

    # Összes CSV fájl keresése
    csv_pattern = os.path.join(directory, "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print("Nincs CSV fájl a mappában!")
        return None, None

    # Timestamp kinyerése a fájlnevekből
    file_timestamps = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        name_without_ext, extension = os.path.splitext(filename)
        timestamp = name_without_ext.split("_")[3:5]
        full_timestamp = f"{timestamp[0]}_{timestamp[1]}"
        file_timestamps.append((file_path, full_timestamp))

    if not file_timestamps:
        print("Egyetlen fájlban sem található érvényes timestamp!")
        return None, None

    # Legutolsó fájl kiválasztása timestamp alapján
    latest_file, latest_timestamp = max(file_timestamps, key=lambda x: x[1])

    print(f"\nLegutolsó fájl: {os.path.basename(latest_file)}")
    print(f"Timestamp: {latest_timestamp}")

    return latest_file, latest_timestamp