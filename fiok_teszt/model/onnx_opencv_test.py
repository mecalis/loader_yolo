import cv2
import numpy as np

# --- PARAMÉTEREK ---
model_path = "C:/ML/loader_yolo/fiok_teszt/model/model_2025-08-25.onnx"
image_path = "C:/ML/loader_yolo/fiok_teszt/test_images/Frame_20250626_114442_062_LoaderFS1010FK_StorageHangerFaceplateStateDetect_Right.jpg"
conf_threshold = 0.78
nms_threshold = 0.2
input_size = 640   # YOLOv8/YOLOv11 tipikus inputméret

# --- MODELL BETÖLTÉSE ---
net = cv2.dnn.readNet(model_path)

# --- KÉP BETÖLTÉSE ---
image = cv2.imread(image_path)
orig_h, orig_w = image.shape[:2]

# --- ELŐFELDOLGOZÁS ---
blob = cv2.dnn.blobFromImage(image,
                             scalefactor=1/255.0,
                             size=(input_size, input_size),
                             swapRB=True,
                             crop=False)
net.setInput(blob)

# --- FORWARD ---
outputs = net.forward()   # shape: (1, N, 85)  [x,y,w,h,obj_conf,class1,...]
outputs = np.squeeze(outputs)  # -> (N, 85)

# --- POSZTFELDOLGOZÁS ---
boxes = []
confidences = []
class_ids = []

for det in outputs:
    scores = det[5:]  # class scorek
    print(f"scores: {scores}")
    class_id = np.argmax(scores)
    print(f"class_id: {class_id}")
    confidence = det[4] * scores[class_id]
    print(f"det4: {det[4]}")

    if confidence > conf_threshold:
        # YOLO koordináták (középpont x,y,w,h) → bal-felső x,y,w,h
        cx, cy, w, h = det[0], det[1], det[2], det[3]
        x = int((cx - w/2) * orig_w / input_size)
        y = int((cy - h/2) * orig_h / input_size)
        w = int(w * orig_w / input_size)
        h = int(h * orig_h / input_size)

        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)
        #print(f"classid: {class_ids}, boxok: {[x, y, w, h]}, conf: {confidence}")

# NMS (Non-Maximum Suppression)
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# --- EREDMÉNYEK KIRAJZOLÁSA ---
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = f"ID:{class_ids[i]} {confidences[i]:.2f}"
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    print(f"{label}, boxes: {boxes[i]}")

# Megjelenítés
cv2.imshow("Detekció", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
