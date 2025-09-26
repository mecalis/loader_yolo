'''
3. CSV-ben tárolódó eredmények kiértékelését végző script.
Az "érdekes" képeket külön mappába másolja további felhasználásra.
'''

import pandas as pd
import numpy as np
import os
import glob
import shutil
from fiok_train_helper_functions import find_latest_csv_by_timestamp

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)



#df = pd.read_csv("yolo_detection_results_20250707_142233.csv")
latest_file, latest_timestamp = find_latest_csv_by_timestamp()
df = pd.read_csv(latest_file)

print("-"*100)
print(f"A képek, amikben error szerepel: {df["error"].count()}")
df_error = df[df["error"].notna()]
print(df_error.head())

pivot = pd.crosstab(df["filename"], df["class_id"])
pivot.columns = ["faceplate", "atg"]
df_pivot_no_faceplate = pivot[pivot["faceplate"]<1]
df_pivot_no_atg = pivot[pivot["atg"]<1]
print("-"*100)
print(f"\nAzoknak a képeknek a száma, ahol nem talált faceplatet: {df_pivot_no_faceplate.shape[0]}")
print(df_pivot_no_faceplate)
print("-"*100)
print(f"Azoknak a képeknek a száma, ahol nem talált atgt: {df_pivot_no_atg.shape[0]}")
print(df_pivot_no_atg)

df_images_filter = (pivot["faceplate"] != 1) | (pivot["atg"] != 1)
df_images = pivot[df_images_filter]

print("-"*100)
print("\nÉrdekes képek:")
#print(df_images)

script_dir = os.path.dirname(os.path.abspath(__file__))
df_images_path = os.path.join(script_dir, 'erdekes_kepek')
test_images_path = os.path.join(script_dir, 'test_images')

if os.path.exists(df_images_path):
    try:
        shutil.rmtree(df_images_path)
    except Exception as e:
        print(e)
os.makedirs(df_images_path, exist_ok=True)

def copy_images(image_name, source=test_images_path, dest=df_images_path):
    shutil.copy2(os.path.join(source, image_name), os.path.join(dest, image_name))

for image_name in df_images.index:
    copy_images(image_name)

print(f"#INFO: Másolás kész: {len(df_images.index)} db")

df_images = df_images.reset_index()
df_images['file_name_without_ext'] = df_images['filename'].str.split('.').str[0]
df_images[['frame', 'date', 'time', 'number', 'loader', 'job', 'side']] = df_images['file_name_without_ext'].str.split('_', n=6, expand=True)

print("-"*100)
print(df_images)
print("-"*100)
print("\nFaceplate szerinti megoszlás:")
print(df_images["faceplate"].value_counts())

print("-"*100)
print("\nATG szerinti megoszlás:")
print(df_images["atg"].value_counts())

print("-"*100)
print("\nLoaderek szerinti megoszlás:")
print(df_images["loader"].value_counts())

df_images.to_csv("fiok_teszt/eredmeny.csv")






