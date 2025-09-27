# YOLO Loader Project

  Ez a repository egy YOLO-alapú élkereső rendszerhez készült, amely a modell betanításától kezdve a tesztelésen át az eredmények kiértékeléséig és exportálásáig tartalmaz Python scripteket.
  A logisztikai központban raktárrobotok, úgynevezett Loaderek, végzik a termékek ki és betárolását a raktárban. A pontos működéshez a robotok mozgását kamerás rendszerek segítik, melyeket hagyomásos, élkeresés alapú megoldással programozták. A Loaderek által generált összes hiba ~40%-a, három különböző, kamerás részfolyamathoz köthető. A hibákat súlyosbítja, hogy kamerás hiba esetén a folyamat során ütközés, vagy egyéb, a vezérlés által nem javítható hiba jelentkezik. Ilyenkor a berendezés leáll az üzemeltető technikus manuális hibajavításáig.
  A fentiek miatt alakítottam ki a depp learning alapú YOLO object detection megoldást, ami egy nagyságrenddel nagyobb stabilitással végzi a feladatát.
  A kamerás mérő folyamat során meg kell határozni a vízszintes tartóléc alsó élének és a nagy fióklemez oldalsó élének metszéspontjában lévő koordinátát, melynek a kép közepétől mért távolsága adja meg a robot korrekciós offset értékét.
  Az alábbi képen mind a 3 koordináta rendszer ábrázolásra került:
  - a kép közepét jelző koordináta rendszer zöld színnel,
  - a meglévő kamerás script kimenete piros színnel,
  - az objektumkeresés eredményét felhasználva az új script által generált koordináta rendszer kék színnel.

A Loader mindig az adott polchely egy dedikált, névleges koordinátájánál áll meg, majd a kamerás feldolgozással korrigáljuk a beállás pontatlanságát. A pontatlanság adódik egyrészt a négy keréken guruló eszköz beállási pontosságától, másrészt a helyenként különböző tömegekkel terhelt polcrendszer lehajlásától és vetemedésétől.

## Tartalomjegyzék
- [Telepítés](#telepítés)
- [Használat](#használat)
- [Scriptek](#scriptek)
  - [Betanító script - (1-fiok_train.py)](https://github.com/mecalis/loader_yolo/blob/main/1-fiok_train.py)
  - [Tesztelő script - (2-fiok_teszt.py)](https://github.com/mecalis/loader_yolo/blob/main/2-fiok_teszt.py)
  - [Eredménykiértékelő script (3-fiok_csv_eredmenyek.py)](https://github.com/mecalis/loader_yolo/blob/main/3-fiok_csv_eredmenyek.py)
  - [CSV feldolgozó script (4-generate_text_file.py)](https://github.com/mecalis/loader_yolo/blob/main/4-generate_text_file.py)
  - [Teszt-összehasonlító script (5-fiok_teszt_eredmeny_osszehasonlito.py)](https://github.com/mecalis/loader_yolo/blob/main/5-fiok_teszt_eredmeny_osszehasonlito.py)
  - [Modell export script (6-fiok_model_export.py)](https://github.com/mecalis/loader_yolo/blob/main/6-fiok_model_export.py)
  - [ONNX tesztelő script (7-onnx_teszt.py)](https://github.com/mecalis/loader_yolo/blob/main/7-onnx_teszt.py)

---

## Telepítés
A futtatáshoz Python 3.12 szükséges.

Követelmények telepítése:
```bash
pip install -r requirements.txt
```

## Scriptek
1. Betanító script (1-fiok_train.py)
Létrehozza a tanító és teszt adathalmazokat, a könyvtárszerkezetet, betanítja a modellt.

2. Tesztelő script (2-fiok_teszt.py)
A betanított modellen futtatja a teszt adathalmaz összes, 1200+ db képét. Predikciókat készít és statisztikákat gyűjt. Az eredményeket .csv formátumban menti.

3. Eredménykiértékelő script (3-fiok_csv_eredmenyek.py)
A .csv fájlban tárolt eredményeket feldolgozza. Minden betanítás után futtatandó. Megvizsgálja a model pontosságát aszerint, hogy minden képen megtalálta-e a model az objektumokat.
Az „érdekes” képeket külön mappába másolja további elemzéshez. 

4. CSV feldolgozó script (4-generate_text_file.py)
Létrehozza a results.csv fájlt az alábbi formátumban: fájlnév, x_tengely, y_tengely. Ezek az értékek lesznek az korrigációs eltolásvektorok a robot verélése számára.


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
