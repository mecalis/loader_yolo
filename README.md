# YOLO Loader Project

  Ez a repository egy YOLO-alapú élkereső rendszerhez készült, amely a modell betanításától kezdve a tesztelésen át az eredmények kiértékeléséig és exportálásáig tartalmaz Python scripteket.
  A logisztikai központban raktárrobotok, úgynevezett Loaderek, végzik a termékek ki és betárolását a raktárban. A pontos működéshez a robotok mozgását kamerás rendszerek segítik, melyeket hagyomásos, élkeresés alapú megoldással programozták. A Loaderek által generált összes hiba ~40%-a, három különböző, kamerás részfolyamathoz köthető. A hibákat súlyosbítja, hogy kamerás hiba esetén a folyamat során ütközés, vagy egyéb, a vezérlés által nem javítható hiba jelentkezik. Ilyenkor a berendezés leáll az üzemeltető technikus manuális hibajavításáig.
  A fentiek miatt alakítottam ki a depp learning alapú YOLO object detection megoldást, ami egy nagyságrenddel nagyobb stabilitással végzi a feladatát.

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
```

Scriptek
1. Betanító script (1-fiok_train.py)
Létrehozza a tanító és teszt adathalmazokat, a könyvtárszerkezetet, betanítja a modellt.

2. Tesztelő script (2-fiok_teszt.py)
A betanított modellen futtatja a teszt adathalmaz összes, 1200+ db képét. Predikciókat készít és statisztikákat gyűjt. Az eredményeket .csv formátumban menti.

3. Eredménykiértékelő script (evaluate_results.py)

A .csv fájlban tárolt eredményeket feldolgozza

Az „érdekes” képeket külön mappába másolja további elemzéshez

4. CSV feldolgozó script (process_csv.py)

Létrehozza a results.csv fájlt az alábbi formátumban:

fájlnév, x_tengely, y_tengely


Működés:

Képnevek kigyűjtése

ATG és faceplate detekciók feldolgozása

ATG esetén: alsó koordináták átlagolása

Faceplate esetén: bal/jobb oldalfüggő koordináta választás

Az eredményt results.csv fájlban menti

5. Teszt-összehasonlító script (compare_results.py)

Összehasonlítja a hagyományos loader élkereső és a YOLO által számolt tengelyeket

Az eredményeket hisztogramon ábrázolja

±15 pixeles eltérésű detekciókat külön DataFrame-ben tárolja (df_erdekesek)

A kijelölt képekre vizualizáció kerül:

zöld: névleges x és y tengely

kék: YOLO által számolt tengelykereszt

piros: hagyományos élkereső tengelykereszt

Az 1226 tesztképből 46 lett kiemelve, az első 9 képből 3×3 mozaik készül

6. Modell export script (export_model.py)

A betanított modell exportálása további felhasználásra

7. ONNX tesztelő script (onnx_test.py)

Az exportált ONNX modell futtatása és ellenőrzése
