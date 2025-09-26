"""
2. Teszt script.
A betanított modell alapján a teszt adathalmaz összes képén végighaladva predikál és kigyűjti a statisztikákat.
Az eredményt .csv fájlba menti.
"""

import os
import pandas as pd
from PIL import Image
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import cv2
import shutil

class loader_fiok_teszter:
    """
    YOLO object detection tesztelő osztály Faceplate és ATG objektumok detektálására
    """

    def __init__(self, yolo_model_path=None, test_images_path=None, result_save=False):
        """
        Inicializálás

        Args:
            yolo_model_path (str, optional): YOLO modell fájl elérési útja.
                                           Ha None, akkor a 'model' mappában keresi.
            test_images_path (str, optional): Teszt képek mappájának elérési útja.
                                            Ha None, akkor a 'test_images' mappát használja.
            result_save (bool): Ha True, akkor CSV-be menti az eredményeket
        """
        # Script futtatási könyvtár meghatározása
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Alapértelmezett útvonalak beállítása
        self.yolo_model_path = self._get_model_path(yolo_model_path)
        self.test_images_path = self._get_test_images_path(test_images_path)
        self.result_save = result_save

        # Osztály nevek
        self.class_names = {0: 'Faceplate', 1: 'ATG'}

        # Eredmények tárolása
        self.results_df = None
        self.detection_results = []

        # YOLO modell betöltése
        self.model = self._load_yolo_model()

        # Teszt képek betöltése
        self.test_images = self._load_test_images()

        print(f"Loader Fiók Teszter inicializálva:")
        print(f"- YOLO modell: {self.yolo_model_path}")
        print(f"- Teszt képek: {self.test_images_path}")
        print(f"- Talált képek száma: {len(self.test_images)}")
        print(f"- Eredmények mentése: {result_save}")

    def _get_model_path(self, yolo_model_path):
        """
        YOLO modell elérési útjának meghatározása

        Args:
            yolo_model_path (str or None): Megadott modell útvonal

        Returns:
            str: Végleges modell útvonal
        """
        if yolo_model_path is None:
            # Alapértelmezett: model mappa a script mellett
            model_dir = os.path.join(self.script_dir, 'model')

            # Keresés .pt fájlokra a model mappában
            if os.path.exists(model_dir):
                pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
                if pt_files:
                    model_path = os.path.join(model_dir, pt_files[0])
                    print(f"Automatikusan talált modell: {pt_files[0]}")
                    return model_path
                else:
                    raise Exception(f"Nem található .pt fájl a model mappában: {model_dir}")
            else:
                raise Exception(f"A model mappa nem található: {model_dir}")
        else:
            # Megadott útvonal használata
            if not os.path.exists(yolo_model_path):
                raise Exception(f"A megadott modell fájl nem található: {yolo_model_path}")
            return yolo_model_path

    def _get_test_images_path(self, test_images_path):
        """
        Teszt képek mappájának elérési útjának meghatározása

        Args:
            test_images_path (str or None): Megadott teszt képek útvonal

        Returns:
            str: Végleges teszt képek útvonal
        """
        if test_images_path is None:
            # Alapértelmezett: test_images mappa a script mellett
            default_path = os.path.join(self.script_dir, 'test_images')
            if not os.path.exists(default_path):
                raise Exception(f"A test_images mappa nem található: {default_path}")
            return default_path
        else:
            # Megadott útvonal használata
            if not os.path.exists(test_images_path):
                raise Exception(f"A megadott teszt képek mappa nem található: {test_images_path}")
            return test_images_path

    def _load_yolo_model(self):
        """YOLO modell betöltése"""
        try:
            model = YOLO(self.yolo_model_path)
            print(f"YOLO modell sikeresen betöltve: {self.yolo_model_path}")
            return model
        except Exception as e:
            raise Exception(f"Hiba a YOLO modell betöltésekor: {e}")

    def _load_test_images(self):
        """Teszt képek betöltése a mappából"""
        images = []
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        if not os.path.exists(self.test_images_path):
            raise Exception(f"A teszt képek mappája nem található: {self.test_images_path}")

        for filename in os.listdir(self.test_images_path):
            if filename.lower().endswith(supported_extensions):
                file_path = os.path.join(self.test_images_path, filename)
                images.append({
                    'filename': filename,
                    'path': file_path
                })

        if not images:
            raise Exception(f"Nem található kép a megadott mappában: {self.test_images_path}")

        return images

    def _clahe(self, image):
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) alkalmazása.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray_image)
        return cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

    def _draw_boxes(self, image_copy, results):
        #image_copy = cv2.imread(image).copy()
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

    def run_detection(self):
        """
        YOLO object detection futtatása az összes teszt képen
        """
        print("\nObject detection futtatása...")

        result_images_path = os.path.join(self.script_dir, 'result_images')
        print("#INFO: result_images_path:", result_images_path)
        try:
            shutil.rmtree(result_images_path)
        except:
            print("#INFO: Nem volt result_images mappa, amit törölni lehetett volna.")

        try:
            os.makedirs(result_images_path, exist_ok=True)
        except:
            print("#INFO: result_images mappát megpróbálta létrehozni, de nem sikerült.")


        for i, image_info in enumerate(self.test_images):
            #print(f"Feldolgozás: {image_info['filename']} ({i+1}/{len(self.test_images)})")
            if i%100 == 0:
                print(f"Feldolgozás: ({i+1}/{len(self.test_images)})")
            try:
                # YOLO detection futtatása
                loaded_image = cv2.imread(os.path.join(self.script_dir, 'test_images', image_info['filename']))
                image_to_detect = self._clahe(loaded_image)

                #results = self.model(image_info['path'], conf = 0.7, iou = 0.95, verbose=False)
                results = self.model(image_to_detect, conf=0.78, iou=0.2, verbose=False)

                copy_image = self._draw_boxes(image_to_detect, results)
                image_to_write = os.path.join(result_images_path, os.path.basename(image_info['path']))
                cv2.imwrite(image_to_write, copy_image)
                # Eredmények feldolgozása
                detection_results = self._process_detection_result(
                    image_info['filename'],
                    results[0]
                )

                # Minden detekció hozzáadása a listához
                self.detection_results.extend(detection_results)

            except Exception as e:
                print(f"Hiba a {image_info['filename']} feldolgozásakor: {e}")
                # Hiba esetén is rögzítjük az eredményt
                error_result = {
                    'filename': image_info['filename'],
                    'detection_id': 0,
                    'class_id': None,
                    'class_name': 'Error',
                    'confidence': 0.0,
                    'x1': None,
                    'y1': None,
                    'x2': None,
                    'y2': None,
                    'center_x': None,
                    'center_y': None,
                    'width': None,
                    'height': None,
                    'area': None,
                    'error': str(e)
                }
                self.detection_results.append(error_result)

        # DataFrame létrehozása
        self._create_dataframe()

        # Eredmények mentése, ha szükséges
        if self.result_save:
            self._save_results()

        print(f"\nDetection befejezve! Összesen {len(self.detection_results)} kép feldolgozva.")
        return self.results_df

    def _process_detection_result(self, filename, result):
        """
        Egy kép detection eredményének feldolgozása
        Minden detekció egy külön dict-et ad vissza

        Args:
            filename (str): Kép fájlneve
            result: YOLO detection eredmény

        Returns:
            list: Feldolgozott eredmények listája (minden detekció egy dict)
        """
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                # Koordináták kinyerése (xyxy formátum)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Bounding box középpontja és méretei
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                area = width * height

                detection = {
                    'filename': filename,
                    'detection_id': i + 1,
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, f'Unknown_{class_id}'),
                    'confidence': round(confidence, 4),
                    'x1': round(x1, 2),
                    'y1': round(y1, 2),
                    'x2': round(x2, 2),
                    'y2': round(y2, 2),
                    'center_x': round(center_x, 2),
                    'center_y': round(center_y, 2),
                    'width': round(width, 2),
                    'height': round(height, 2),
                    'area': round(area, 2),
                    'error': None
                }

                detections.append(detection)
        else:
            # Ha nincs detekció, akkor is létrehozunk egy sort
            no_detection = {
                'filename': filename,
                'detection_id': 0,
                'class_id': None,
                'class_name': 'No_Detection',
                'confidence': 0.0,
                'x1': None,
                'y1': None,
                'x2': None,
                'y2': None,
                'center_x': None,
                'center_y': None,
                'width': None,
                'height': None,
                'area': None,
                'error': None
            }
            detections.append(no_detection)

        return detections

    def _create_dataframe(self):
        """Pandas DataFrame létrehozása az eredményekből"""
        self.results_df = pd.DataFrame(self.detection_results)

        # Oszlopok sorrendje
        columns_order = [
            'filename',
            'detection_id',
            'class_id',
            'class_name',
            'confidence',
            'x1', 'y1', 'x2', 'y2',
            'center_x', 'center_y',
            'width', 'height', 'area',
            'error'
        ]

        self.results_df = self.results_df[columns_order]

        print(f"\nDataFrame létrehozva {len(self.results_df)} detektálással")

        # Összesítő statisztikák
        total_images = self.results_df['filename'].nunique()
        total_detections = len(self.results_df[self.results_df['class_name'] != 'No_Detection'])
        faceplate_count = len(self.results_df[self.results_df['class_name'] == 'Faceplate'])
        atg_count = len(self.results_df[self.results_df['class_name'] == 'ATG'])
        avg_confidence = self.results_df[self.results_df['confidence'] > 0]['confidence'].mean()

        print(f"\nÖsszesítés:")
        print(f"- Feldolgozott képek: {total_images}")
        print(f"- Összes detektálás: {total_detections}")
        print(f"- Faceplate detektálások: {faceplate_count}")
        print(f"- ATG detektálások: {atg_count}")
        print(f"- Átlagos confidence: {avg_confidence:.4f}" if not pd.isna(avg_confidence) else "- Átlagos confidence: N/A")

    def _save_results(self):
        """Eredmények mentése CSV fájlba"""
        if self.results_df is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"yolo_detection_results_{timestamp}.csv"
            # CSV fájlt a script könyvtárába mentjük
            csv_path = os.path.join(self.script_dir, csv_filename)

            self.results_df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"\nEredmények mentve: {csv_path}")
        else:
            print("\nNincs eredmény a mentéshez!")

    def get_results(self):
        """Eredmények DataFrame visszaadása"""
        return self.results_df

    def print_summary(self):
        """Összesítő statisztikák kiírása"""
        if self.results_df is not None:
            print("\n" + "="*60)
            print("YOLO DETECTION ÖSSZESÍTŐ")
            print("="*60)

            total_images = self.results_df['filename'].nunique()
            total_detections = len(self.results_df[self.results_df['class_name'] != 'No_Detection'])
            faceplate_count = len(self.results_df[self.results_df['class_name'] == 'Faceplate'])
            atg_count = len(self.results_df[self.results_df['class_name'] == 'ATG'])
            no_detection_count = len(self.results_df[self.results_df['class_name'] == 'No_Detection'])
            error_count = len(self.results_df[self.results_df['error'].notna()])

            print(f"Feldolgozott képek: {total_images}")
            print(f"Összes detektálás: {total_detections}")
            print(f"Képek detektálás nélkül: {no_detection_count}")
            print(f"Hibák: {error_count}")

            print(f"\nDetektált objektumok:")
            print(f"- Faceplate: {faceplate_count}")
            print(f"- ATG: {atg_count}")

            # Confidence statisztikák
            valid_confidences = self.results_df[self.results_df['confidence'] > 0]['confidence']
            if len(valid_confidences) > 0:
                print(f"\nConfidence statisztikák:")
                print(f"- Átlagos: {valid_confidences.mean():.4f}")
                print(f"- Minimum: {valid_confidences.min():.4f}")
                print(f"- Maximum: {valid_confidences.max():.4f}")

            # Képenkénti statisztikák
            print(f"\nKépenkénti bontás:")
            image_stats = self.results_df.groupby('filename').agg({
                'class_name': lambda x: (x != 'No_Detection').sum(),
                'confidence': 'mean'
            }).round(4)
            image_stats.columns = ['detections_count', 'avg_confidence']
            print(image_stats.head(10))

            print("="*60)
        else:
            print("Nincs eredmény a megjelenítéshez!")


if __name__ == "__main__":
    # Alapértelmezett használat (automatikus útvonalak)
    teszter = loader_fiok_teszter(result_save=True)

    # Vagy egyedi útvonalakkal
    # teszter = loader_fiok_teszter(
    #     yolo_model_path="custom/path/to/model.pt",
    #     test_images_path="custom/path/to/images",
    #     result_save=True
    # )

    # Detection futtatása
    results_df = teszter.run_detection()

    # Összesítő megjelenítése
    teszter.print_summary()

    # Eredmények megtekintése
    print("\nEredmények DataFrame:")
    print(results_df.head())