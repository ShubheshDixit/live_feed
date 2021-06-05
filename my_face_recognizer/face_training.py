import cv2
import numpy as np
from PIL import Image
import os

def train_faces():
    dataset_path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # function to get the images and label data
    def getImagesAndLabels(dataset_path):
        imagePaths = []
        for f in os.listdir(dataset_path):
            imagePaths.extend([os.path.join(dataset_path, f, fr) for fr in os.listdir(os.path.join(dataset_path, f))])
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples, ids
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(dataset_path)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') 
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained".format(len(np.unique(ids))))

if __name__ == '__main__':
    train_faces()