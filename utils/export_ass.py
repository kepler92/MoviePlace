import os
import time


def getTimeStamp(ms):
    return time.strftime("%H:%M:%S.", time.gmtime(ms / 1000)) + str(ms % 1000)[:2]


__ass_header__ = [
        "[Script Info]\n",
        "; This is a Sub Station Alpha v4 script.\n",
        "; For Sub Station Alpha info and downloads,\n",
        "; go to http://www.eswat.demon.co.uk/\n",
        "Title: TEST\n",
        "Original Script: RoRo\n",
        "Script Updated By: version 2.8.01\n",
        "ScriptType: v4.00\n",
        "Collisions: Normal\n",
        "PlayResX: 1280\n",
        "PlayResY: 720\n",
        "PlayDepth: 0\n",
        "Timer: 100,0000\n",
        "  \n",
        "[V4 Styles]\n",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, TertiaryColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, AlphaLevel, Encoding\n",
        "Style: AudioObject, Arial,28,11861244,11861244,11861244,-2147483640,-1,0,1,1,2,2,30,30,30,0,0\n",
        "Style: VisualObject, Arial,28,11842812,11842812,11842812,-2147483640,-1,0,1,1,2,5,0,0,0,90,0\n",
        "  \n",
        "[Events]\n",
        "Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    ]


class ExportAss:
    def __init__(self, video_name, subtitle_name=None):
        if subtitle_name is None:
            subtitle_name = os.path.splitext(video_name)[0] + ".ass"
        self.__subtitle_file = open(subtitle_name, "w")
        global __ass_header__
        self.__subtitle_file.writelines(__ass_header__)

    def write_datum(self, start_time, end_time, label_name, label_prob, log=False):
        if len(label_name) == 0:
            return

        start_time = getTimeStamp(int(start_time * 1000))
        end_time = getTimeStamp(int(end_time * 1000))

        label_text = ""
        for idx in range(min(len(label_name), len(label_prob))):
            label_text = label_text + str(label_name[idx]) + "(" + str(round(float(label_prob[idx]), 2)) + ") "
        label_text = label_text[:-1]
        ao_string = "Dialogue: Marked=0, %s,%s,AudioObject, NTP,0000,0000,0000,,%s\n" \
                      % (start_time, end_time, label_text)
        self.__subtitle_file.writelines(ao_string)
        if log:
            print ao_string

    def __del__(self):
        self.__subtitle_file.close()
