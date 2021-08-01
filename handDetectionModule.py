import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False, max_hands=2, detection_con=0.5,track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        #getting the hands trained data from mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands,self.detection_con,self.track_con)
        self.mpDraw = mp.solutions.drawing_utils  
    def findHands(self, img, draw=True):
        #converting into black and white
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #processing the blk & wht data
        self.results = self.hands.process(imgRGB)

        #making the marks for hand visible
        if self.results.multi_hand_landmarks:
            for handLMK in self.results.multi_hand_landmarks:
                if draw:
                    #drawing landmarks
                    self.mpDraw.draw_landmarks(img, handLMK, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum=0,draw=True):
        lnList =[]

        if self.results.multi_hand_landmarks:
            myHands = self.results.multi_hand_landmarks[handNum]

            for id,ln in enumerate(myHands.landmark):
                #print(id,ln)
                #getting height,width,channel
                h,w,c = img.shape
                #making them int
                cx, cy = int(ln.x*w),int(ln.y*h)
                #print(id,cx,cy)
                lnList.append([id,cx,cy])
                if draw:
                        #drawing circle on index and thumb
                    cv2.circle(img, (cx,cy), 8, (234,255,0), cv2.FILLED )
        return lnList


    

def main():
    cam = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        ptime = time.time()
        #reading the videocapture
        success, img = cam.read()
        img = detector.findHands(img)
        lnList = detector.findPosition(img)

        if len(lnList) != 0:
            print(lnList[4])

        ctime= time.time()
        fps=1/(ctime-ptime)
        ptime=ctime

        cv2.putText(img, "fps="+str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)













if __name__ == "__main__":
    main()