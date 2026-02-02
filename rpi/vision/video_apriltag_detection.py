import cv2
import apriltag
import os
import numpy as np 


# Capture parameters
CAMERA_ID = 0  # Camera ID (usually 0 for built-in webcam)

cam_name = 'usb_cam'

#path = './calib_images/image'
path = '/home/ale/projects/vision/apriltag_images/April_Image'

cap = cv2.VideoCapture(CAMERA_ID)
num = 0

options = apriltag.DetectorOptions(families="tag25h9")
detector = apriltag.Detector(options)

tag_size = 147 # mm

camera_params = [1.307934937365716678e+03, 1.307934937365716678e+03, 
                 2.992942985625898018e+02, 1.922528980697034910e+02]

dst_coeff = [-1.060483605864858542e-01, 
              5.994140458346827849e+00,
              5.640448388169525332e-03, 
             -8.635100527637183665e-03,
             -8.210441861892326187e+01]

def _draw_pose_axes(image, pose_matrix, camera_params, dst_coeff, tag_size):
    """Draw 3D coordinate axes on the detected tag"""
    
    # Define 3D points for the axes (in tag frame)
    # Origin at tag center, axes extend by half the tag size
    axis_length = tag_size / 2
    opoints = np.array([
        [0, 0, 0],              # origin
        [axis_length, 0, 0],    # x-axis point
        [0, axis_length, 0],    # y-axis point
        [0, 0, -axis_length]    # z-axis point (into camera)
    ]).reshape(-1, 1, 3)
    
    # Camera intrinsic matrix
    fx, fy, cx, cy = camera_params
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Extract rotation and translation from pose matrix
    rvec, _ = cv2.Rodrigues(pose_matrix[:3, :3])
    tvec = pose_matrix[:3, 3]
    
    # Project 3D points to image plane
    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, np.array(dst_coeff))
    ipoints = np.round(ipoints).astype(int).reshape(-1, 2)
    
    # Draw axes
    origin = tuple(ipoints[0])
    cv2.line(image, origin, tuple(ipoints[1]), (0, 0, 255), 3)    # X-axis (red)
    cv2.line(image, origin, tuple(ipoints[2]), (0, 255, 0), 3)    # Y-axis (green)
    cv2.line(image, origin, tuple(ipoints[3]), (255, 0, 0), 3)    # Z-axis (blue)
    
while cap.isOpened():

    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)

    cv2.imshow('Camera', img)

    # loop over the AprilTag detection results
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        # draw the bounding box of the AprilTag detection
        cv2.line(img, ptA, ptB, (0, 255, 80), 2)
        cv2.line(img, ptB, ptC, (0, 255, 80), 2)
        cv2.line(img, ptC, ptD, (0, 255, 80), 2)
        cv2.line(img, ptD, ptA, (0, 255, 80), 2)

        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)

        # draw the tag family on the image
        tagId = str(r.tag_id)
        text = "tag_id : " + tagId

        cv2.putText(img, text, (ptA[0] + 20, ptA[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 2)
        
        pose_matrix, init_error, final_error = detector.detection_pose(
            r,
            camera_params,
            tag_size=tag_size,
            z_sign=1
        )
        
        # pose_matrix is a 4x4 transformation matrix
        _draw_pose_axes(img, pose_matrix, camera_params, dst_coeff, tag_size)

    k = cv2.waitKey(5)

    if k == 27: 
        break 
    elif k == ord('s'): 
        cv2.imwrite(path + str(num) + '.png', img)
        print('Image ' + str(num) + ' saved')
        num +=1


    cv2.imshow('AprilTag', img)

cap.release()
cv2.destroyAllWindows()