import cv2
import traceback
from cvzone.HandTrackingModule import HandDetector
from util import overlay_transparent

cap = cv2.VideoCapture(0)

cap.set(3, 1288)
cap.set(4, 728)

detector = HandDetector(detectionCon=0.8)
startDist = None
scale = 0
cx, cy = 500, 500

bit_mask = ((1 << 33) - 2)


def isHandGesture(hand1, hand2):
    return detector.fingersUp(hand1) == [1, 1, 1, 1, 1] and \
           detector.fingersUp(hand2) == [1, 1, 1, 1, 1]


def resizeImg(foreground, background):
    h1, w1, _ = foreground.shape

    newH, newW = ((h1 + scale) & bit_mask), ((w1 + scale) & bit_mask)

    foreground = cv2.resize(foreground, (newW, newH))
    return overlay_transparent(background, foreground, cx - (newW >> 1), cy - (newH >> 1))


while True:
    success, video_img = cap.read()
    hands, ig = detector.findHands(video_img)
    img = cv2.imread("img/rasengan.png", cv2.IMREAD_UNCHANGED)
    if len(hands) == 2:
        left_hand = hands[0]
        right_hand = hands[1]
        if isHandGesture(left_hand, right_hand):
            if startDist is None:
                length, info, video_img = detector.findDistance(left_hand["center"], right_hand["center"], video_img)
                startDist = length
            length, info, video_img = detector.findDistance(left_hand["center"], right_hand["center"], video_img)
            scale = int((length - startDist) // 2)
            cx, cy = info[4:]
            print(scale)
        try:
            video_img = resizeImg(img, video_img)
        except Exception as e:
            traceback.print_exc()
    else:
        startDist = None

    cv2.imshow("Image", video_img)
    key = cv2.waitKey(1)
    if key & 0XFF == ord(' '):
        break
