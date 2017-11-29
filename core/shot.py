import os
import scenedetect

import cv2
try:
    from cv2 import cv2
except ImportError:
    pass


detector_list = [ scenedetect.detectors.ContentDetector() ]


def get_shot_list(path, capture=None, log=False):
    shot_list_file_path = os.path.splitext(path)[0] + ".shotlist"
    if os.path.isfile(shot_list_file_path):
        with open(shot_list_file_path, "r") as shot_list_file:
            shot_list = [int(x) for x in shot_list_file.read().split()]
    else:
        if capture is None:
            shot_list = __detect_shot_list_by_path(path=path)
        else:
            shot_list = __detect_shot_list(capture=capture)
        with open(shot_list_file_path, "w") as shot_list_file:
            for item in shot_list:
                shot_list_file.write("{0}\n".format(item))
    if log:
        print shot_list
    return shot_list


def __detect_shot_list_by_path(path):
    scene_list = []
    video_framerate, frames_read = scenedetect.detect_scenes_file(path, scene_list, detector_list, quiet_mode=True)
    scene_list = scene_list + [int(frames_read)]
    return scene_list


def __detect_shot_list(capture):
    scene_list = []
    frames_read = scenedetect.detect_scenes(capture, scene_list, detector_list, quiet_mode=True)
    scene_list = scene_list + [int(frames_read)]
    capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
    return scene_list