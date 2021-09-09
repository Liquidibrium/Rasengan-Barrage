import cv2
import traceback
from cvzone.HandTrackingModule import HandDetector
from util import overlay_transparent

BIT_MASK = ((1 << 33) - 2)


def is_hand_gesture(detector, hand1, hand2, up_down=True):
    fingers_list = [1, 1, 1, 1, 1] if up_down else [0, 0, 0, 0, 0]
    return detector.fingersUp(hand1) == fingers_list and detector.fingersUp(hand2) == fingers_list


def draw_transparent(foreground, background, cx, cy, newW, newH):
    return overlay_transparent(background, foreground, cx - (newW >> 1), cy - (newH >> 1))


def get_resized_values(foreground, scale):
    h1, w1, _ = foreground.shape
    new_h, new_w = ((h1 + scale) & BIT_MASK), ((w1 + scale) & BIT_MASK)
    foreground = cv2.resize(foreground, (new_w, new_h))
    return new_h, new_w, foreground


def main_func():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(detectionCon=0.8)
    start_dist = None
    scale = 0
    cx, cy = 500, 500
    img = cv2.imread("img/rasengan.png", cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    success, video_img = cap.read()  # first read for height and width
    h, w, c = video_img.shape  # can be made consts
    while True:
        success, video_img = cap.read()
        video_img = cv2.flip(video_img, 1)
        # hands = detector.findHands(video_img, draw=False)  # don't draw
        hands, video_img = detector.findHands(video_img)  # draw
        if len(hands) == 2:
            left_hand = hands[0]
            right_hand = hands[1]
            if is_hand_gesture(detector, left_hand, right_hand):
                length, info = detector.findDistance(left_hand["center"], right_hand["center"])
                if start_dist is None:
                    start_dist = length
                scale = int((length - start_dist) // 2)
                cx, cy = info[4:]
            try:
                new_h, new_w, foreground = get_resized_values(img, scale)
                start_x = cx - new_w // 2
                start_y = cy - new_h // 2
                end_x = start_x + new_w
                end_y = start_y + new_h
                if not (start_x < 0 or start_y < 0 or end_x > w or end_y > h):
                    video_img = draw_transparent(foreground, video_img, cx, cy, new_w, new_h)
            except ValueError:
                traceback.print_exc()
        else:
            start_dist = None
        cv2.imshow("Image", video_img)
        key = cv2.waitKey(1)
        if key & 0XFF == ord(' '):
            break


if __name__ == "__main__":
    main_func()
