from ultralytics import YOLO
import os

model_name = "best_faceplate_0_atg_1.pt"

model_path = os.path.join(os.getcwd(), 'fiok_teszt', 'model', model_name)

print(f"#INFO: Model betöltése: {model_path}")
model = YOLO(model_path)

# 2️⃣ Export ONNX formátumba
model.export(
    format="onnx",           # export formátum
    opset=12,                # ONNX opset verzió (ált. 12 jó)
    dynamic=False,            # dinamikus input méretek engedélyezése
    simplify=True,           # egyszerűsített ONNX graf, ne írd át!!
    imgsz=(640, 640),
    #nms=True
)

print("✅ YOLOv11 modell sikeresen exportálva ONNX formátumba!")

