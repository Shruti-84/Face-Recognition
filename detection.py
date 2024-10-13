import cv2
import numpy as np
from os import listdir
from os.path import isfile , join

data_path = "C:/Users/user/Desktop/dataset/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Lables = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images,dtype = np.uint8))
    Lables.append(i)

Lables = np.asarray(Lables, dtype =np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Lables))

print("Dataset model training completed ")

face_classifier = cv2.CascadeClassifier('C:/Users/user/Desktop/coding 1/face recognization/haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3,5)
    
    if face is():
        return img,[]
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w, y+h), (0,255,0),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi,(200,200))
        
        return img, roi
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # image, face = face_detector(frame)
    
    
    try:
        face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
        
        if confidence > 82:
            cv2.putText(frame, "Known Face", (250,450), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            cv2.imshow('face cropper', frame)
            
        else:
            cv2.putText(frame,"unknown",(250,450), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            cv2.imshow('Face cropper', frame)
    except:
        cv2.putText(frame, "Face not found", (250,450),1,(255,0,0),2)
        # cv2.putText(frame,"unknown",(250,450), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('face cropper', frame)
        pass
    if cv2.waitKey(1) ==13:
        break
    
cap.release()
cv2.destroyAllWindows()
