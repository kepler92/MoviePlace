from __future__ import print_function
import sys

from core import shot, object, place
from utils.argv_parser import *

import cv2
try:
    from cv2 import cv2
except ImportError:
    pass


if __name__ == "__main__":
    argv = argv_parser(sys.argv)

    cap = cv2.VideoCapture()
    cap.open(argv[0])
    video_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    frame_count = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    frame_move = video_fps

    shot_list = shot.get_shot_list(path=argv[0], capture=cap, log=True)
    shot_idx = 0

    video_width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    video_second = 1

    object_list = ['person']
    object_filter = object.Object(width=video_width, height=video_height,
                                  object_list=object_list, object_max_threshold=0.35,
                                  object_sum_number=5, obj_sum_threshold=0.5)

    place_list = []
    place_classifier = place.Place()

    while cap.isOpened():
        frame_number = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        frame_second = frame_number / video_fps

        res, frame = cap.read()
        if res is False:
            break

        #cv2.imwrite("{0}.jpg".format(frame_number), frame)

        detect_flag, detect_size = object_filter.detect(frame)
        if detect_flag is False:
            place_result_idx, place_result_prob = place_classifier.classifier(frame)
            #print (place_result_idx)

        frame_next = video_second * frame_move
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_next)

        if frame_next >= shot_list[shot_idx]:
            shot_end = int(shot_list[shot_idx] - 1)
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, shot_end)

            res, frame = cap.read()
            if res is False:
                exit(-1)

            #cv2.imwrite("{0}.jpg".format(shot_end), frame)

            detect_flag, detect_size = object_filter.detect(frame)
            if detect_flag is False:
                place_result_idx, place_result_prob = place_classifier.classifier(frame)

            place_result_idx, place_result_prob = place_classifier.estimate(shot_idx + 1)

            if frame_next == shot_list[shot_idx]:
                video_second += 1
            shot_idx += 1

        else:
            video_second += 1



