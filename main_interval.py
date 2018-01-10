from __future__ import print_function
from utils.argument_parser import *

from core import shot, objects, place
from utils import export_ass, compression

import cv2
try:
    from cv2 import cv2
except ImportError:
    pass


if __name__ == "__main__":
    video_name = args.video_name

    cap = cv2.VideoCapture()
    cap.open(video_name)

    if not cap.isOpened():
        raise Exception("Video file does not open.")

    video_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    frame_count = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    frame_move = video_fps / 2.

    video_width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    video_second = 1

    object_list = ['person']
    object_filter = objects.Objects(width=video_width, height=video_height,
                                    object_list=object_list, object_max_threshold=0.35,
                                    object_sum_number=5, obj_sum_threshold=0.5)

    place_list = []
    place_classifier = place.PlaceWindow(5, 0.8)

    place_ass = export_ass.ExportAss(video_name)

    while cap.isOpened():
        frame_number = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        frame_second = frame_number / video_fps

        res, frame = cap.read()
        frame = compression.jpeg(frame)
        if res is False or frame_number == frame_count:
            break

        print(frame_number)

        detect_flag, detect_size = object_filter.detect(frame)
        if detect_flag is False:
            place_result_idx, place_result_prob = place_classifier.classifier(frame)

        place_result_idx, place_result_prob = place_classifier.estimate()
        place_result_label = place.get_label_name(place_result_idx)

        frame_next = int(round(video_second * frame_move))

        shot_start = frame_second
        shot_end = frame_next / video_fps

        place.print_place('', shot_start, shot_end, place_result_idx, place_result_label, place_result_prob)
        place_ass.write_datum(shot_start, shot_end, place_result_label, place_result_prob)

        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_next)
        video_second += 1