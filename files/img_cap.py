import cv2
from pathlib import Path
from time import sleep
import json
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

video_capture = cv2.VideoCapture(0)

username = input("Enter Your Name: ")
ID = input("Enter your ID: ")
emailId = input("Enter your emailId: ")

count = 1


def saveimage(img, user_Id, user_name, imageId): 
    """
    save a image in specific folder
    """
    Path("files\dataset/{}".format(user_name)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite("files\dataset/{}/{}_{}.jpg".format(user_name, user_Id, imageId), gray)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for count in range(11):
        saveimage(gray, ID, username, count)
        count += 1
        print(f"[INFO] Taking Photo = {count}")
        sleep(0.2)
    else:
        sleep(0.3)
        video_capture.release()
        cv2.destroyAllWindows()
        break

video_capture.release()
cv2.destroyAllWindows()
