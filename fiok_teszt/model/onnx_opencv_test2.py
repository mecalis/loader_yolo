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

blob = cv2.dnn.blobFromImage(image, 1 / 255, (640, 640), [0, 0, 0])
net.setInput(blob)

#új

# Kimenet
outs = net.forward(net.getUnconnectedOutLayersNames())
print("Output shape:", outs[0].shape)

# Első néhány detekció vizsgálata
for i, det in enumerate(outs[0][:5]):  # csak az első 5 sor
    print(f"\nDetekció {i}:")
    print("  teljes hossz:", len(det))
    print("  raw értékek (első 10):", det[:10])

    # bounding box + confidence + class rész
    x, y, w, h, conf = det[:5]
    scores = det[5:]
    print("  scores shape:", scores.shape)
    print("  max score:", np.max(scores))
    print("  class id (argmax):", np.argmax(scores))


#új vége


output_layers = net.getUnconnectedOutLayersNames()
outputs = net.forward(output_layers)

# Lists to hold respective values while unwrapping.
class_ids = ['frist class', ' second class']
confidences = []
boxes = []

# Rows.
rows = outputs[0].shape[1]

image_height, image_width = image.shape[:2]

# Resizing factor.
x_factor = image_width / 640  # get var value from your loaded image
y_factor = image_height / 640

# Iterate through 25200 detections.
for r in range(rows):
    row = outputs[0][0][r]
    confidence = row[4]

    # Discard bad detections and continue.
    if confidence >= conf_threshold:
        classes_scores = row[5:]

        # Get the index of max class score.
        class_id = np.argmax(classes_scores)

        #  Continue if the class score is above threshold.
        if (classes_scores[class_id] > nms_threshold):
            confidences.append(confidence)
            class_ids.append(class_id)

            cx, cy, w, h = row[0], row[1], row[2], row[3]

            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            box = np.array([left, top, width, height])
            boxes.append(box)

# Perform non maximum suppression to eliminate redundant overlapping boxes with
# lower confidences.
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.78, nms_threshold)
for i in indices:
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    cv2.rectangle(image, (left, top), (left + width, top + height), (255,0,0), 3 * 1)
    label = "{}:{:.2f}".format(class_ids[i], confidences[i])
    #draw_label(input_image, label, left, top)
    print(label)  # will contain label and confidence