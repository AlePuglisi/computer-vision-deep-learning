import cv2
import os


# Capture parameters
CAMERA_ID = 2  # Camera ID (usually 0 for built-in webcam)

cam_name = 'gopro_hero3'
os.path.join()

cap = cv2.VideoCapture(CAMERA_ID)
num = 0

while cap.isOpened():

    success, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27: 
        break 
    elif k == ord('s'): 
        cv2.imwrite('calib_images/img' + str(num) + '.png', img)
        print('Image ' + str(num) + ' saved')
        num +=1

    cv2.imshow('Camera', img)

cap.release()
cv2.destroyAllWindows()

