'''이미지 용량이 200kb 이상인지 확인하는 코드'''

import os
import cv2
from PIL import Image
import shutil
import math

input_path = "./image_downloaded/pikachu/"
image_paths = []

move_path = "./image_downloaded/pikachu/resized/"

if os.path.isdir(input_path):
    # os.listdir : 이 디렉토리에 있는 전체 파일의 이름을 리스트 형태로 반환한다
    for inp_file in os.listdir(input_path):
        image_paths += [input_path + inp_file]

    # 그중에서 jpg, png, jpeg 확장자를 가진 파일만 남긴다
image_paths = [inp_file for inp_file in image_paths
               if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

# the main loop
for image_path in image_paths:

    filename = image_path.split('/')[-1]

    size = os.path.getsize(image_path)

    size_kb = math.floor(size/1024)

    if size_kb > 200:
        print(filename + "'s size ="+str(size_kb))
        # shutil.move(image_path, move_path+filename)
        # print("file "+filename+" is moved. size = "+str(size_kb))

    # foo = Image.open("path\\to\\image.jpg")
    # size_origin = foo.size

