import cv2
import traceback
import imageio
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from util import overlay_transparent
from itertools import cycle

BIT_MASK = ((1 << 65) - 2)
DEFAULT_PNG = "img/rasengan0.png"
DEFAULT_GIF = "img/rasengan0.gif"
DEFAULT_MP4 = "video/Rasengan-green-screen.mp4"


def get_gif_frames(path_to_gif: str):
    frames = []
    im = imageio.get_reader(path_to_gif)
    for frame in im:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frames.append(np.array(frame))
    return frames


def get_png(path_to_png: str):
    img = cv2.imread(path_to_png, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    return [img]


def get_mp4_frames(path_to_mp4):
    cap = cv2.VideoCapture(path_to_mp4)
    frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        u_green = np.array([100, 255, 100, 255])
        l_green = np.array([0, 150, 0, 255])
        mask = cv2.inRange(frame, l_green, u_green)
        frame[mask != 0] = np.array([0, 0, 0, 0])
        frames.append(frame)
    return frames


def is_hand_gesture(detector, hand1, hand2, up_down=True):
    fingers_list = [1, 1, 1, 1, 1] if up_down else [0, 0, 0, 0, 0]
    return detector.fingersUp(hand1) == fingers_list and detector.fingersUp(hand2) == fingers_list


def draw_transparent(foreground, background, cx, cy, new_w, new_h):
    return overlay_transparent(background, foreground, cx - (new_w >> 1), cy - (new_h >> 1))


def get_resized_values(foreground, scale):
    h1, w1, _ = foreground.shape
    new_h, new_w = (max(h1 + scale, 2) & BIT_MASK), (max(w1 + scale, 2) & BIT_MASK)
    foreground = cv2.resize(foreground, (new_w, new_h))
    return new_h, new_w, foreground


def get_frames_to_render(file_type: str, path_to_file: str):
    if file_type == "png":
        if not path_to_file:
            path_to_file = DEFAULT_PNG
        return get_png(path_to_file)
    elif file_type == "gif":
        if not path_to_file:
            path_to_file = DEFAULT_GIF
        return get_gif_frames(path_to_file)
    elif file_type == "mp4":
        if not path_to_file:
            path_to_file = DEFAULT_MP4
        return get_mp4_frames(path_to_file)
    raise NotImplemented


def capture_live(frames):
    cap = cv2.VideoCapture(0)  # windows warning solution  cv2.CAP_DSHOW
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(detectionCon=0.8)
    start_dist = None
    scale = 0
    cx, cy = 500, 500

    success, video_img = cap.read()  # first read for height and width
    h, w, c = video_img.shape  # can be made consts
    while cap.isOpened():
        success, video_img = cap.read()
        video_img = cv2.flip(video_img, 1)
        # hands = detector.findHands(video_img, draw=False)  # don't draw
        hands, video_img = detector.findHands(video_img)  # draw
        if len(hands) == 2:
            img = next(frames)
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
                cx = max(cx, new_w // 2)
                cy = max(cy, new_h // 2)
                video_img = draw_transparent(foreground, video_img, cx, cy, new_w, new_h)
            except ValueError:
                traceback.print_exc()
        else:
            start_dist = None
        cv2.imshow("Image", video_img)
        key = cv2.waitKey(1)
        if key & 0XFF == ord(' '):
            cap.release()
            break


def live_video_generator(file_type, path_to_file=None):
    frames = get_frames_to_render(file_type, path_to_file)
    capture_live(cycle(frames))


if __name__ == "__main__":
    # extract_frames(DEFAULT_MP4)
    live_video_generator("mp4")
    # get_png(DEFAULT_PNG)
