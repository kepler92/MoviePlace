from __future__ import print_function
from utils.argument_parser import *

from core import shot, place
from utils import export_ass, compression

import cv2
try:
    from cv2 import cv2
except ImportError:
    pass


if __name__ == "__main__":
    video_name = args.video_name
    gpu_id = args.gpu_id

    cap = cv2.VideoCapture()
    cap.open(video_name)

    if not cap.isOpened():
        raise Exception("Video file does not open.")

    video_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    frame_count = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    frame_move = video_fps

    shot_list = shot.get_shot_list(path=video_name, capture=cap)
    shot_idx = 1

    video_width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    video_second = 1

    # object_list = ['person']
    # object_filter = objects.Objects(width=video_width, height=video_height,
    #                                 object_list=object_list, object_max_threshold=0.35,
    #                                 object_sum_number=5, obj_sum_threshold=0.5)

    place_list = []
    place_classifier = place.PlaceShot()

    place_ass = export_ass.ExportAss(video_name)
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 7460)

    while cap.isOpened():
        frame_number = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        frame_second = frame_number / video_fps

        res, frame = cap.read()
        frame = compression.jpeg(frame)
        if res is False or frame_number == frame_count:
            break

        print (frame_number)
        place_result_idx, place_result_prob = place_classifier.classifier(frame)

        if frame_number >= 7465:
            break