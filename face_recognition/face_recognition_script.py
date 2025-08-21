import cv2
import os
import numpy as np

data_path = 'C:/Users/ACER/Desktop/face_recognition_project/face_recognition'  
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = []  
labels = []
label_map = {}
label_counter = 0

for filename in os.listdir(data_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(data_path, filename)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_rect = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_rect:
            face = gray[y:y+h, x:x+w]
            name = os.path.splitext(filename)[0]
            if name not in label_map:
                label_map[name] = label_counter
                label_counter += 1
            faces.append(face)
            labels.append(label_map[name])

recognizer.train(faces, np.array(labels))
print("✔ Обучението приключи.")

test_img = cv2.imread('C:/Users/ACER/Desktop/face_recognition_project/face_recognition/test.jpg')

gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
faces_rect = face_cascade.detectMultiScale(gray_test, 1.3, 5)

for (x, y, w, h) in faces_rect:
    face = gray_test[y:y+h, x:x+w]
    label, confidence = recognizer.predict(face)
    name = [k for k, v in label_map.items() if v == label][0]
    
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(test_img, f"{name} ({round(100 - confidence)}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

scale_percent = 30  
width = int(test_img.shape[1] * scale_percent / 100)
height = int(test_img.shape[0] * scale_percent / 100)
dim = (width, height)

resized_test_img = cv2.resize(test_img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Резултат', resized_test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
