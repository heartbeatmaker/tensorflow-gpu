import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import requests

from urllib.request import urlopen
import urllib.request
import socket
import threading
import shutil
import re

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

from time import sleep




# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'test1.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
# PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
# PATH_TO_IMAGE = "./test_images/0002.jpg"

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')



'''서버에 메시지를 보낸다'''
def sendMsg(message):
    print("sending msg to server. message="+message)
    socket.send(message.encode())  # 문자열을 인코딩해야한다


'''서버로부터 메시지를 수신받는다'''
def msgReceivingThread():
    print("msgReceivingThread starts")
    while 1:
        data = socket.recv(65535)
        message_received = data.decode()
        print("received data from Server : ", message_received)

        # 서버로부터 이미지 이름을 받으면, input 폴더에 해당 이미지를 저장한다
        if message_received.split("/")[0] == "start":
            print("start message")

            image_name_origin = message_received.split("/")[-1]
            image_name_without_extension = image_name_origin.split(".jpg")[0]

            image_name = image_name_without_extension+".jpg"

            print("image_name="+image_name)

            image_url = "http://15.164.193.65/pikachu_images/"+image_name

            urllib.request.urlretrieve(image_url, "./predict_input/"+image_name)
            print("Downloaded: "+image_name)


socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.connect(('15.164.193.65', 8000))

# 입장메시지를 보낸다. id와 username, 가입한 방 id를 보내는 것임
tt = threading.Thread(target=sendMsg, args=("connect/10000/pikachu/10000"+"\n",))
tt.start()


# 클라이언트의 메인 쓰레드는 서버로부터 온 데이터를 읽어들이는 것을 반복한다
t = threading.Thread(target=msgReceivingThread)
t.start()


while True:

    image_paths = []
    input_path = "./predict_input/"

    # input 디렉토리에서 사진파일을 가져온다
    # os.listdir : 이 디렉토리에 있는 전체 파일의 이름을 리스트 형태로 반환한다

    if os.path.isdir(input_path):
        # os.listdir : 이 디렉토리에 있는 전체 파일의 이름을 리스트 형태로 반환한다
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]

        # 그중에서 이미지 확장자를 가진 파일만 남긴다
    image_paths = [inp_file for inp_file in image_paths
                   if (inp_file[-4:] in ['.jpg', '.png', 'JPEG', 'jpeg', 'PNG'])]

    if len(image_paths) > 0:

        print("Found "+str(len(image_paths))+" images")
        index = 0

        sleep(0.5)

        for image_path in image_paths:

            index += 1
            filename = image_path.split('/')[-1]

            print("image "+str(index)+" : "+image_path)

            try:
                # Load image using OpenCV and
                # expand image dimensions to have shape: [1, None, None, 3]
                # i.e. a single-column array, where each item in the column has the pixel RGB value
                image = cv2.imread(image_path)
                image_expanded = np.expand_dims(image, axis=0)

                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})

                print("predict done")
                number_of_pikachu = 0

                if len(boxes) > 0:

                    # Draw the results of the detection (aka 'visulaize the results')
                    '''정확도가 0.5 이상이면 박스를 그린다
                    피카츄의 개수를 반환한다'''
                    (image, number_of_pikachu) = vis_util.visualize_boxes_and_labels_on_image_array_and_return_number(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.60)

                print("number of pikachu="+str(number_of_pikachu))

                # 분석한 파일을 output 폴더에 저장한다
                filename_output = "processed_"+str(number_of_pikachu)+"_"+filename

                cv2.imwrite("./predict_output/"+filename_output, np.uint8(image))
                print("saved processed file")

                try:
                    url = "http://15.164.193.65/pikachu_fileUpload.php"
                    file = {'uploaded_file': open("./predict_output/"+filename_output, 'rb')}
                    r = requests.post(url, files=file)

                    response = r.text
                    print("file upload) response from server: "+response)

                    if response == "success":
                        sendMsg("pikachu_server/success/"+filename_output+"\n")
                    else:
                        sendMsg("pikachu_server/fail/"+filename+"\n")

                except Exception as error:
                    print("file upload error: "+str(error))
                    sendMsg("pikachu_server/fail/"+filename+"\n")

                # 원본 파일을 다른 폴더로 옮긴다
                shutil.move(image_path, "./predict_original_images/"+filename)

            except Exception as err:
                print("error:"+str(err))

                # 클라이언트에게 실패 메시지를 발송한다
                sendMsg("pikachu_server/fail/"+filename+"\n")

                # 원본 파일을 다른 폴더로 옮긴다
                shutil.move(image_path, "./predict_original_images/"+filename)

        print("This loop ended")



