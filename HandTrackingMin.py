import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # This is to open the camera
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # hand method
mpDraw = mp.solutions.drawing_utils  # By this we can have lines drawn

pTime = 0  # for fps
cTime = 0  # for fps

while True:
    success, img = cap.read()  # To open the camera
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # To open the camera
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 20, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # inside first for loop. HAND_CONNECTIONS
            # helps to draw landmarks
    cTime = time.time()
    fps = 1 / (cTime - pTime)  # fps calculation
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # font, thickness, color

    cv2.imshow("image", img)
    cv2.waitKey(1)  #
