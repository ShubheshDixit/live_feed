import cv2
import os, time
from tqdm import tqdm

def get_face_data():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # For each person, enter one numeric face id
    face_id = input('\n enter user id end press <return> ==>  ')
    user_name = input('\n enter user name end press <return> ==>  ')
    print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    file_count = 0
    pbar = tqdm(total=360)
    while(True):
        ret, img = cam.read()
        # img = cv2.flip(img, -1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20)
        )
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            file_count += 1
            pbar.update(1)
            os.chdir(os.getcwd())
            if not os.path.exists(f'dataset/{str(face_id)}'):
                os.mkdir(f'dataset/{str(face_id)}')
            cv2.imwrite(f"dataset/{str(face_id)}/{user_name}." + str(face_id) + '.' +  
                        str(file_count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            pbar.close()
            break
        elif count >= 360: # Take 360 face sample and stop video
            pbar.close()
            break
            
if __name__ == '__main__':
    get_face_data()
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()