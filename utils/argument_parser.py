import argparse
from utils.setting import set_gpu_id
import os


class CustomAction(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 nargs=nargs,
                                 const=const,
                                 default=default,
                                 type=type,
                                 choices=choices,
                                 required=required,
                                 help=help,
                                 metavar=metavar,
                                 )
        return

    def __call__(self, *args, **kwargs):
        pass


class ActionFile(CustomAction):
    @staticmethod
    def __get__extension(path):
        return os.path.splitext(path)[1].lower().split('.')[1]

    def __call__(self, parser, namespace, values, option_string=None):
        video_ext = ['mp4', 'avi']
        if not os.path.isfile(values):
            parser.error("File does not exist.")
        elif self.__get__extension(values) not in video_ext:
            parser.error("Unsupported video extension. Supported Only " + ", ".join(video_ext))
        setattr(namespace, self.dest, values)


class ActionGPU(CustomAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = str(values)
        set_gpu_id(values)
        setattr(namespace, self.dest, values)


parser = argparse.ArgumentParser()
parser.add_argument('video_name', action=ActionFile, help='Video file name', type=str, metavar='video_name')
parser.add_argument('-g', action=ActionGPU, dest='gpu_id', help='GPU ID', type=int, metavar="gpu_id")
args = parser.parse_args()
