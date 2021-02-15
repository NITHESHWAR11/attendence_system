import cv2
import os
import numpy as np
from PIL import Image
import pickle

basic_dir = os.path.dirname(os.path.abspath(__file__))
Image_dir = os.path.join(basic_dir, "dataset")
face_cascate = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
recognition = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_label = []
x_train = []
print("[INFO] TRAINING.....")
print("[INFO] Converting into an array")
for root, dirs, files in os.walk(Image_dir):
    for file in files:
        if file.endswith("jpeg") or file.endswith("jpg"):
            path = os.path.join(root, file)
            print(path)
            label = os.path.basename(root).replace(" ", "_").lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id+=1
            id_ = label_ids[label]
            pil_image =Image.open(path).convert("L")
            size = (550, 550)
            final = pil_image.resize(size, Image.ANTIALIAS)
            image_Array = np.array(final, "uint8")
            faces = face_cascate.detectMultiScale(image_Array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_Array[y:y+h, x:x+w]
                x_train.append(roi)
                y_label.append(id_)
with open("label.pickle", "wb") as f:
    pickle.dump(label_ids, f)
recognition.train(x_train, np.array(y_label))
recognition.save("trainner.yml")
print("[INFO] Success.. ")
print("[INFO] Training finished")
