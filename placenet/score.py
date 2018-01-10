import caffe
import numpy as np
import scipy
from scipy.misc import imresize
from skimage import transform
import os

import cv2
try:
    from cv2 import cv2
except ImportError:
    pass


rootdir = os.getcwd()
os.chdir(os.path.dirname( os.path.abspath( __file__ ) ))


MODEL_FILE = './deploy.prototxt'
PRETRAINED = './vgg16_places365.caffemodel'
#MODEL_FILE = './deploy_381.prototxt'
#PRETRAINED = './fc7_100_iter_300000.caffemodel'
imagenet_labels_filename = './categories_places365.txt'


with open(imagenet_labels_filename, "r") as f:
    lines = f.readlines()
    lines_len = len(lines)
    label_list = [None for x in range(lines_len)]
    for line in lines:
        label_catergory, label_id = line.split()
        label_name = label_catergory.split("/")[-1]
        label_list[int(label_id)] = label_name


caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)


os.chdir(rootdir)


def detect(image, net):
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.cv.CV_INTER_LINEAR)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.
    img = imresize(img, [224, 224])
    img = img.astype(np.uint8)
    img = img.transpose(-1, 0, 1)
    net.forward_all(data=np.asarray([img]))

    prob = net.blobs['prob'].data[0]
    return np.reshape(prob, (1, prob.shape[0]))


if __name__ == "__main__":
    data_set = raw_input("folder : ")
    files = os.listdir(data_set)
    files = np.sort(files)

    for cc, x in enumerate(files):
        print "x = ",x
        file_id = x.split('.')[0]
        print "file_id = ",file_id
        image_path = data_set + '/' + file_id
        print "image_path =", image_path
        images =os.listdir(image_path)
        print "iamges = ",images
        for c, y in enumerate(images):
            image_id = y.split('.')[0]
            image_id_txt = image_path + '/' + image_id #+ '.txt'
            image_id = image_path + '/' + image_id +'.jpg'
            print "image_id = ", image_id
            image_id_path = image_path +  '/'
            print "image_id_path = ",image_id_path
            __detect(image_id, net, image_id_path, image_id_txt)
    
