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

5. Fiók teszt eredmény összehasonlító script (5-fiok_teszt_eredmeny_osszehasonlito.py)
A script feldolgozza a meglévő kamerás algoritmus által 1200 darab képre lefuttatott kimenetet. Összehasonltja az eredményeket, kiemeli az eltéréseket további vizsgálatra.

6. Model exportáló script (6-fiok_model_export.py)

7. ONNX tesztelő script
A c++-ban írt vezérlés számára kiexportált .onnx formátumú neurális háló modelljének visszanyitása az opencv DNN moduljával. Ellenőrizni szükséges, hogy valóban kompatibilis-e és megnyitható-e a Loadereken lévő opencv verzióval.

