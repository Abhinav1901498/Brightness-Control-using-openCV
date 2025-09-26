# Importing Libraries
import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np

# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

Draw = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read video frame by frame
    _, frame = cap.read()

    # Flip image
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB image
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    Process = hands.process(frameRGB)

    landmarkList = []
    # if hands are present in image(frame)
    if Process.multi_hand_landmarks:
        # detect handmarks
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                # store height and width of image
                height, width, color_channels = frame.shape

                # calculate and append x, y coordinates
                # of handmarks from image(frame) to lmList
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])

            # draw Landmarks
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    # If landmarks list is not empty
    if landmarkList != []:
        # store x,y coordinates of (tip of) thumb
        x_1, y_1 = landmarkList[4][1], landmarkList[4][2]

        # store x,y coordinates of (tip of) index finger
        x_2, y_2 = landmarkList[8][1], landmarkList[8][2]

        # draw circle on thumb and index finger tip
        cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)

        # draw line from tip of thumb to tip of index finger
        cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)

        # ✅ calculate distance between thumb & index
        L = hypot(x_2 - x_1, y_2 - y_1)

        # Map distance to brightness range (0–100)
        b_level = np.interp(L, [15, 220], [0, 100])

        # set brightness
        sbc.set_brightness(int(b_level))

        # 🔹 show brightness level on screen
        cv2.putText(frame, f'Brightness: {int(b_level)} %',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 0, 0), 3)

        # 🔹 (Optional) progress bar
        cv2.rectangle(frame, (50, 150), (300, 180), (255, 255, 255), 2)
        cv2.rectangle(frame, (50, 150), (50 + int(2.5 * b_level), 180), (0, 255, 0), -1)

    # Display Video
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
