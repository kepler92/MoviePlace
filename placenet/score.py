import caffe
import numpy as np
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

#save_top = open("place_scene.txt",'w')

################ save top_5 ####################################
'''
def top_5(image_id, labels, out, top_k):
    for i in range(0, 5):    
        image_str = "{}".format(image_id)
        save_top.write(image_str)
        label_str = ", 0, 0, 0, 0, {}".format(labels[top_k[i]]) #0, 0, 0, 0,#0, 28, 0, 0,#0,56,0,0
        save_top.write(label_str)
    	probs = out['prob'][0][top_k[i]]
   	probs_str = ", {}\n".format(probs)
    	save_top.write(probs_str)
    	print ""
    save_top.write('\n')
'''

################ extract score ####################################


def detect(image, net):
    img = cv2.resize(image, dsize=(224, 224), interpolation=cv2.cv.CV_INTER_LINEAR)
    img = img.transpose(-1, 0, 1)
    net.forward_all(data=np.asarray([img]))
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #img = imresize(img, [224, 224])
    # net.forward_all(data=np.asarray([img.transpose(2, 0, 1)]))
    #prob = net.blobs['prob'].data[0]
    prob = net.blobs['prob'].data[0]
    return np.reshape(prob, (1, prob.shape[0]))


def __detect(image_id, net, iamge_id_path, image_id_txt):
    #count = count + 1
    #fc8_1 = []
    #save = open(image_id_txt,'w')
    
    #for i in range(len(image_id)):
    #img = caffe.io.load_image(image_id)
    #img = imresize(img, [224, 224])
    #img = img.astype(np.uint8)

    img = cv2.imread(image_id)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.cv.CV_INTER_LINEAR)
    #img = imresize(img, [224, 224])

    print "img = ", img.shape

    out = net.forward_all(data=np.asarray([img.transpose(2, 0, 1)]))

    #labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\s')
    #top_k = net.blobs['prob'].data[0].flatten().argsort()[-1: -381 : -1]
    
    #fc8 = net.blobs['prob'].data
    #label_prob = out['prob'][0][top_k[0]]
    #print label_prob
    #fc8_1.append(fc8[0])
    #print "fc8 = ",fc8[0].shape
    
    #image_str = "{}".format(image_id)
    #save.write(image_str)
    #probs_str = ", {}\n".format(fc8_1)
    #save.write(probs_str)
    #save.close()

    #np.save(str(image_id_txt), fc8_1)   #save *.npy
    #top_5(image_id, labels, out, top_k)  #extract top_5

    return net.blobs['prob'].data[0]


################ test_npy  ####################################

################ main ####################################
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
    
