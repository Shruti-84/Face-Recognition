import cv2
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier('C:/Users/user/Desktop/coding 1/face recognization/haarcascade_frontalface_default.xml')

def face_extractor(img):

# Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangle around the faces
    if len(faces) == 0 :
        return None

    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cropped_face = img[y:y+h, x:x+w]
    
    return cropped_face

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count = count+1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        file_name_path = "C:/Users/user/Desktop/dataset/" + str(count) + '.jpg'
        
        cv2.imwrite(file_name_path, face)
        
        cv2.putText(frame ,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper', frame)
    else:
        print('face not found')
        pass
    
    if cv2.waitKey(1) == 13 or count == 100:
        break
    
cap.release()
cv2.destroyAllWindows()
    
print('Data set collection completed')


