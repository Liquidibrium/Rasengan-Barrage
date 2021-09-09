import cv2
import traceback
from cvzone.HandTrackingModule import HandDetector
from util import overlay_transparent

BIT_MASK = ((1 << 33) - 2)


def isHandGesture(detector, hand1, hand2, up_down=True):
    fingers_list = [1, 1, 1, 1, 1] if up_down else [0, 0, 0, 0, 0]
    return detector.fingersUp(hand1) == fingers_list and \
           detector.fingersUp(hand2) == fingers_list


def resizeImg(foreground, background, scale, cx, cy):
    h1, w1, _ = foreground.shape

    newH, newW = ((h1 + scale) & BIT_MASK), ((w1 + scale) & BIT_MASK)

    foreground = cv2.resize(foreground, (newW, newH))
    return overlay_transparent(background, foreground, cx - (newW >> 1), cy - (newH >> 1))


def main_func():
    cap = cv2.VideoCapture(0)

    cap.set(3, 1288)
    cap.set(4, 728)

    detector = HandDetector(detectionCon=0.8)
    startDist = None
    scale = 0
    cx, cy = 500, 500

    while True:
        success, video_img = cap.read()
        video_img = cv2.flip(video_img, 1)
        hands, ig = detector.findHands(video_img)
        img = cv2.imread("img/rasengan.png", cv2.IMREAD_UNCHANGED)
        if len(hands) == 2:
            left_hand = hands[0]
            right_hand = hands[1]
            if isHandGesture(detector, left_hand, right_hand):
                length, info = detector.findDistance(left_hand["center"], right_hand["center"])
                if startDist is None:
                    startDist = length

                scale = int((length - startDist) // 2)
                cx, cy = info[4:]
            if not isHandGesture(detector, left_hand, right_hand, False): ## down fingers
                try:
                    video_img = resizeImg(img, video_img, scale, cx, cy)
                except ValueError:
                    traceback.print_exc()
        else:
            startDist = None

        cv2.imshow("Image", video_img)
        key = cv2.waitKey(1)
        if key & 0XFF == ord(' '):
            break


if __name__ == "__main__":
    main_func()
