import cv2
import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw= mp.solutions.drawing_utils

cam = cv2.VideoCapture("D:\pythonfiles\comVisFiles\dancevid.webm")

while True:
    success , img = cam.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,ln in enumerate(result.pose_landmarks.landmark):
            h,w,c = img.shape
            #print(id,ln)
            cx,cy = int(ln.x * w), int(ln.y * h)
            cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)



    cv2.imshow("image",img)
    cv2.waitKey(1)