"""
4. CSV feldolgozó script
Előállítja a fájl név + x tengely + y tengely formátumú results.csv fájlt.

Képek feldolgozása:
1. kép nevek kigyűjtése
2. Végigiterálom a képeken a neveket.
3. Kigyűjtöm az atgket és faceplateket
4. Feldolgozom az atgt-: alsó koordinátát átlagolom
5. Feldolgozom a faceplatet: Right esetében a jobb, Left esetében a bal. Ha több van, akkor először faceplatet választok.

Kimenti a results.csv fájlt
"""

import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# A feldolgozandó csv fájl
path = "fiok_teszt/yolo_detection_results_20250710_161941.csv"

df = pd.read_csv(path)

#Az oldal meghatározása
df['side'] = df['filename'].str.split('.').str[0].str.split('_').str[-1]

'''
Képek feldolgozása:
1. kép nevek kigyűjtése
2. Végigiterálom a képeken a neveket.
3. Kigyűjtöm az atgket és faceplateket
4. Feldolgozom az atgt-: alsó koordinátát átlagolom
5. Feldolgozom a faceplatet: Right esetében a jobb, Left esetében a bal. Ha több van, akkor először faceplatet választok.
'''

pictures = df["filename"].unique()
print("Egyedi képek mennyisége:", len(pictures))
print(df.head())

print("\nEredmények:")
results_df = []

for picture in pictures:
    temp_df = None
    tmp_df_1 = None
    tmp_df_2 = None
    atg_koordinate = None
    faceplate_koordinate = None

    temp_df = df[df["filename"] == picture].reset_index()

    #Ha error volt a képen, adja vissza a None értékekeket, egyébként menjen tovább az if true ágba.
    if temp_df.iloc[0, 4] != "Error":
        #Nincs error, azért mehet a kigyűjtés
        #A detekciókból leszűröm a faceplate-eket -> tmp_df_1
        tmp_df_1 = temp_df[temp_df["class_id"] == 0]
        # A detekciókból leszűröm az atg-ket -> tmp_df_2
        tmp_df_2 = temp_df[temp_df["class_id"] == 1]
        #faceplate feldolgozás
        if tmp_df_1.shape[0] > 0:
            #Ha van faceplate detekció, és "Right" kép -> a bounding boxok jobb alsó koordinátái közül kiválasztom
            # a bal oldalit egy minimum kereséssel. "Left" esetén a jobb oldalit egy maximum kereséssel.
            #Erre azért van szükség, mert egyrészt két, egymás melletti faceplatet is láthat egyszerre,
            #másrészt egyszer a faceplate jobb, máskor a bal oldali függőleges síkja a keresett koordináta. Ez
            #a megoldás kezeli azt a ritka esetet is, ha kettő, szinte teljesen egymást fedő detekció lenne.
            #Ennek az esélyét szabályozta az iou paraméter.
            if tmp_df_1.iloc[0, -1] == "Right":
                faceplate_koordinate = tmp_df_1["x2"].min()
            else:
                faceplate_koordinate = tmp_df_1["x1"].max()

        #atg feldolgozás
        if tmp_df_2.shape[0] > 0:
            # Ha van atg detekció, akkor az alsó vízszintes Y koordinátát átlagolom.
            # Erre azért van szükség, mert egy faceplate mindkét oldalán megtalálja ugyanazt az atgt. Ha ferde az
            # atg, a közepén a magasság pont az átlag lesz.
            atg_koordinate = tmp_df_2["y2"].mean()

    results_df.append({
        "filename": picture, "x": faceplate_koordinate, "y": atg_koordinate
    })
    if picture == 'Frame_20250626_135507_192_LoaderFS1010FK_StorageHangerFaceplateStateDetect_Right.jpg':
        print(picture, faceplate_koordinate, atg_koordinate)


results_df = pd.DataFrame(results_df)
results_df['filename'] = results_df['filename'].str.replace(' ', '')
print(results_df.head())
print("*"*100)
print(results_df.shape)
print("*"*100)
print(results_df.info())
print("*"*100)
print(results_df.describe())
print("*"*100)
print("Ahol None értékek vannak:")
print(results_df[results_df.isnull().any(axis=1)])

results_df.to_csv('fiok_teszt/results.csv', index=False)


