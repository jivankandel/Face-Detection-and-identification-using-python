import os
import imutils
import cv2

image_path = "C:\\Users\\Omen\\Desktop\\AI\\Project\\images\\Sauravchandra"


def save_faces(cascade, imgname):
    img = cv2.imread(os.path.join(image_path, imgname))
    grey=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    for i, face in enumerate(cascade.detectMultiScale(grey)):
        x, y, w, h = face
        sub_face = grey[y:y + h, x:x + w]
        cv2.imwrite(os.path.join("faces", "{}_{}.jpg".format(imgname, i)), sub_face)
       

if __name__ == '__main__':
    face_cascade = "C:\\Users\\Omen\\Desktop\\AI\\Project\\haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(face_cascade)
    # Iterate through files
    for f in [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]:
        save_faces(cascade, f)