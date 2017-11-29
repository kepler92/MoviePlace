from __future__ import print_function
import os

def argv_parser(argv):
    video_name = None
    gpu_number = 0

    argc = len(argv)
    if argc == 1:
        print("Usage: <filename> <gpu>")
        exit(0)

    else:
        if argc >= 2:
            video_name = argv[1]

    return [video_name, gpu_number]


