from __future__ import print_function
import sys
import os


def __parser_error():
    print ("Usage: <filename> <gpu>")
    exit(0)


def __parser(argv):
    video_name = None
    gpu_number = 0

    argc = len(argv)
    if argc == 1:
        __parser_error()

    else:
        if argc >= 2:
            video_name = argv[1]
            if os.path.isfile(video_name) is False:
                __parser_error()

        if argc >= 3:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(argv[2])

    return [video_name, gpu_number]


argv = __parser(sys.argv)