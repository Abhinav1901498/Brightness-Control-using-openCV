import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=1)
Draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Process = hands.process(frameRGB)

    landmarkList = []

    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                h, w, c = frame.shape
                x, y = int(landmarks.x * w), int(landmarks.y * h)
                landmarkList.append([_id, x, y])
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    if landmarkList != []:
        x1, y1 = landmarkList[4][1], landmarkList[4][2]
        x2, y2 = landmarkList[8][1], landmarkList[8][2]

        cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        L = hypot(x2 - x1, y2 - y1)
        b_level = np.interp(L, [15, 220], [0, 100])
        sbc.set_brightness(int(b_level))

        cv2.putText(frame, f'Brightness: {int(b_level)} %',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 0, 0), 3)
        cv2.rectangle(frame, (50, 150), (300, 180), (255, 255, 255), 2)
        cv2.rectangle(frame, (50, 150), (50 + int(2.5 * b_level), 180), (0, 255, 0), -1)

    cv2.imshow('Brightness Control', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
