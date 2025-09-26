# YOLO Loader Project

Ez a repository egy YOLO-alapú élkereső rendszerhez készült, amely a modell betanításától kezdve a tesztelésen át az eredmények kiértékeléséig és exportálásáig tartalmaz Python scripteket.

## Tartalomjegyzék
- [Telepítés](#telepítés)
- [Használat](#használat)
- [Scriptek](#scriptek)
  - [1. Betanító script (`1-fiok_train.py`)](#1-betanító-script-trainpy)
  - [2. Tesztelő script (`2-fiok_teszt.py`)](#2-tesztelő-script-testpy)
  - [3. Eredménykiértékelő script (`3-fiok_csv_eredmenyek.py`)](#3-eredménykiértékelő-script-evaluate_resultspy)
  - [4. CSV feldolgozó script (`4-generate_text_file.py`)](#4-csv-feldolgozó-script-process_csvpy)
  - [5. Teszt-összehasonlító script (`5-fiok_teszt_eredmeny_osszehasonlito.py`)](#5-teszt-összehasonlító-script-compare_resultspy)
  - [6. Modell export script (`6-fiok_model_export.py`)](#6-modell-export-script-export_modelpy)
  - [7. ONNX tesztelő script (`7-onnx_teszt.py`)](#7-onnx-tesztelő-script-onnx_testpy)

---

## Telepítés
A futtatáshoz Python 3.12 szükséges.

Követelmények telepítése:
```bash
pip install -r requirements.txt
