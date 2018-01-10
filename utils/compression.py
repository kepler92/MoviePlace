import cv2
try:
    from cv2 import cv2
except ImportError:
    pass


def jpeg(frame):
    res_enc, frame_enc = cv2.imencode('.jpg', frame)
    frame_dec = cv2.imdecode(frame_enc, cv2.cv.CV_LOAD_IMAGE_COLOR)
    return frame_dec