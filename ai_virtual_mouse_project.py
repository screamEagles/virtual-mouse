import cv2
import numpy as np
import hand_tracking_module as htm
import time
import pyautogui

width_cam, height_cam = 1280, 600
frame_reduction = 50
smoothening = 7

previous_time = 0
previous_location_x, previous_location_y = 0, 0
current_location_x, current_location_y = 0, 0

cap =  cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)
detector = htm.handDetector(maxHands=1)
width_screen, height_screen = pyautogui.size()

while True:
    # 1. find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    landmark_list, bounding_box = detector.findPosition(img)

    # 2. get the tip of the index and middle fingers
    if len(landmark_list) != 0:
        x1, y1 = landmark_list[8][1:]
        x2, y2 = landmark_list[12][1:]
        # print(x1, y1, x2, y2)

        # 3. check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frame_reduction, frame_reduction), (width_cam - frame_reduction, height_cam - frame_reduction), (255, 0, 255), 2)

        # 4. only index finger: moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. convert coordinates
            x3 = np.interp(x1, (frame_reduction, width_cam - frame_reduction), (0, width_screen))
            y3 = np.interp(y1, (frame_reduction, height_cam - frame_reduction), (0, height_screen))

            # 6. smoothen values
            current_location_x = previous_location_x + (x3 - previous_location_x) / smoothening
            current_location_y = previous_location_y + (y3 - previous_location_y) / smoothening

            # 7. move mouse
            pyautogui.moveTo(width_screen - current_location_x, current_location_y)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            previous_location_x, previous_location_y = current_location_x, current_location_y


        # 8. both index and middle fingers are up : clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. find distance between fingers
            length, img, line_info = detector.findDistance(8, 12, img)
            # print(length)
            
            # 10. click mouse if distance short
            if length < 40:
                cv2.circle(img, (line_info[4], line_info[5]), 15, (255, 0, 0), cv2.FILLED)
                pyautogui.click()

    # 11. frame rate
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    (120, 250, 55), 3)

    # 12. display
    # flip it in the end for curiousity
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
