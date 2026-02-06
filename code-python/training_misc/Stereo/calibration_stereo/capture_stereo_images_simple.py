import cv2
import os


# Capture parameters
CAMERA_R_ID = 1  # Camera ID (usually 0 for built-in webcam)
CAMERA_L_ID = 2  # Camera ID (usually 0 for built-in webcam)

cam_name = 'isn'
os.path.join()

capR = cv2.VideoCapture(CAMERA_R_ID)
capL = cv2.VideoCapture(CAMERA_L_ID)
num = 0

while capR.isOpened() and capL.isOpened():

    success, imgR = capR.read()
    success, imgL = capL.read()

    k = cv2.waitKey(5)

    if k == 27: 
        break 
    elif k == ord('s'): 

        cv2.imwrite('calib_images/' + cam_name + '/imgR' + str(num) + '.png', imgR)
        cv2.imwrite('calib_images/' + cam_name + '/imgL' + str(num) + '.png', imgL)

        print('Right Image ' + str(num) + ' saved')
        print('Left Image ' + str(num) + ' saved')

        num +=1

    cv2.imshow('Right Camera', imgR)
    cv2.imshow('Left Camera', imgL)

capR.release()
capL.release()
cv2.destroyAllWindows()

