from __future__ import print_function

from yolo import darknet
import numpy as np
from utils.np_function import argkmax


class object:
    net = darknet.net
    meta = darknet.meta

    def __init__(self, width, height,
                    object_list, object_max_threshold,
                    object_sum_number, obj_sum_threshold):
        self.width = width
        self.height = height
        self.obj_list = object_list
        self.obj_max_thr = object_max_threshold
        self.obj_sum_num = object_sum_number
        self.obj_sum_thr = obj_sum_threshold
        self.obj_max_thr_value = width * height * object_max_threshold * 1.274
        self.obj_sum_thr_value = width * height * obj_sum_threshold * 1.274

    def detect(self, frame, log=None):
        objects = darknet.detect(self.net, self.meta, frame)

        objects_flag = False
        objects_size = []

        for obj_idx, obj_item in enumerate(objects):
            if obj_item[0] in self.obj_list:
                obj_size = obj_item[2][2] * obj_item[2][3]
                objects_size.append(obj_size)

        max_idxs, max_sizes = argkmax(np.array(objects_size))

        if len(max_idxs) != 0:
            max_sizes = max_sizes[:self.obj_sum_num]

            if max_sizes[0] > self.obj_max_thr_value:
                objects_flag = True
            if np.sum(max_sizes) > self.obj_sum_thr_value:
                objects_flag = True

        if log is not None:
            print("#{0}\tObject: {1}".format(log, objects_flag), end=" ")

        return objects_flag, max_sizes

