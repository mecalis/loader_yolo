import os
import numpy as np
import onnx

print("Aktuális munkakönyvtár:", os.getcwd())

class onnx_tester:
    def __init__(self, onnx_model_path = None, test_image = None, result_save = False):
        # Script futtatási könyvtár meghatározása
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Alapértelmezett útvonalak beállítása
        self.onnx_model_path = self._get_model_path(onnx_model_path)
        print(f"#INFO: ONNX modell útvonal: {self.onnx_model_path}")
        self.test_image = self._get_test_image_path(test_image)
        print(f"#INFO: Teszt kép útvonal: {self.test_image}")
        self.result_save = result_save

        # Osztály nevek
        self.class_names = {0: 'Faceplate', 1: 'ATG'}
        # ONNX modell betöltése
        self.model = self._load_onnx_model()
        print("#INFO: Az ONNX nem sérült, használható az opencv dnn moduljához.")

    def _get_model_path(self, onnx_model_name):
        if onnx_model_name is None:
            model_path = os.path.join(self.script_dir, 'fiok_teszt', 'model', 'best_faceplate_0_atg_1.onnx')
            return model_path
        else:
            model_path = os.path.join(self.script_dir, 'fiok_teszt', 'model', onnx_model_name)
            return model_path

    def _load_onnx_model(self):
        """ONNX modell betöltése"""
        try:
            model_path = self.onnx_model_path
            onnx_model = onnx.load(model_path)
            print(f"#INFO: ONNX modell sikeresen betöltve: {self.onnx_model_path}")
            return onnx_model
        except Exception as e:
            raise Exception(f"Hiba az ONNX modell betöltésekor: {e}")

    def _get_test_image_path(self, test_image_path):
        if test_image_path is None:
            image_path = os.path.join(self.script_dir, 'fiok_teszt', 'test_images', "Frame_20250626_114608_064_LoaderFS1010FK_StorageHangerFaceplateStateDetect_Left.jpg")
            return image_path
        else:
            image_path = os.path.join(self.script_dir, 'fiok_teszt', 'test_images', test_image_path)
            return image_path

    def _check_the_onnx_file(self):
        try:
            onnx.checker.check_model(self.model, full_check=True)
        except Exception as e:
            print(f"Az ONNX ellenőrzés hibára futott: {e}")



print("ONNX teszter class:")
onnx_test = onnx_tester(result_save = False)
#model_path = "C:/ML/loader_yolo_p12/fiok_teszt/model/best_faceplate_0_atg_1.onnx"
onnx_model = onnx_test.model

########################################################################################################
#
import cv2
model_path = onnx_test.onnx_model_path
net = cv2.dnn.readNet(model_path)

# Teszt kép betöltése
img = cv2.imread("C:/ML/loader_yolo_p12/fiok_teszt/test_images/Frame_20250626_114608_064_LoaderFS1010FK_StorageHangerFaceplateStateDetect_Left.jpg")
print("A betöltött kép dimenziói:", img.shape)

# Eredeti méret elmentése
h_orig, w_orig = img.shape[:2]
h_in, w_in = 640, 640

# YOLO bemenet előkészítése (640x640, BGR->RGB, 0..1 skálázás)
blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)

# Bemenet beállítása
net.setInput(blob)

# Lefuttatjuk a forward-ot
outs = net.forward() # shape: (1, 6, 8400)
print("Raw output shape:", outs.shape)
# Átalakítás (1, 8400, 6)-ra
outs = np.transpose(outs, (0, 2, 1))
print("Fixed output shape:", outs.shape)
# Most: (1, 8400, 6) → [x,y,w,h,score_cls1,score_cls2]

detections = outs[0]
#print("Első 5 predikció:", outs[0][:5])

####################################################################################################

conf_threshold = 0.3
nms_threshold = 0.4
scale_x = w_orig / w_in
scale_y = h_orig / h_in
boxes = []
scores = []
class_ids = []

for det in detections:
    x, y, w, h = det[:4]
    scores_det = det[4:]
    class_id = np.argmax(scores_det)
    confidence = scores_det[class_id]

    if confidence > conf_threshold:
        # YOLO output: cx, cy, w, h → konvertálás (x1, y1, w, h)
        x1 = int((x - w/2)) #* img.shape[1])
        print("x:", x)
        y1 = int((y - h/2)) #* img.shape[0])

        x1 *= scale_x
        w *= scale_x
        y1 *= scale_y
        h *= scale_y
        x1 = int(x1)
        y1 = int(y1)
        w = int(w)
        h = int(h)
        boxes.append([x1, y1, w, h])
        scores.append(float(confidence))
        class_ids.append(class_id)

print("megtalált boxok:", boxes)
# OpenCV NMS
indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
print(indices)
print(indices.shape)

class_names = ["faceplate", "atg"]
for i in indices:
    #i = i[0]
    box = boxes[i]
    print(box)
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, str(class_names[class_ids[i]]), (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imshow("Detections", img)
cv2.waitKey(0)
