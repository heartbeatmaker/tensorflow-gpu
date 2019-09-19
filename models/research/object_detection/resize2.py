'''이미지 크기를 줄이는 코드'''

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

    foo = Image.open(image_path)

    if foo.mode != "RGB":
        foo = foo.convert("RGB")

    filename = image_path.split('/')[-1]
    filename_no_extension = filename.split('.')[0]

    size_x = foo.size[0]
    size_y = foo.size[1]

    size_x_revised = size_x
    size_y_revised = size_y

    if size_x > 700 or size_y > 700:
        print("size is big")

        while size_x_revised >= 700 or size_y_revised >= 700:
            size_x_revised = size_x_revised*0.8
            size_y_revised = size_y_revised*0.8

            print(filename+" size: "+str(size_x)+","+str(size_y)+" => "+str(size_x_revised)+","+str(size_y_revised))

        print("while end")
        size_x_revised = math.floor(size_x_revised)
        size_y_revised = math.floor(size_y_revised)

        foo.resize((size_x_revised, size_y_revised), Image.ANTIALIAS)
        foo.save(move_path+filename_no_extension+".jpg", optimize=True,quality=95)
        os.remove(image_path)

        print(filename+" is resized and moved. size: "+str(size_x)+","+str(size_y)+" => "+str(size_x_revised)+","+str(size_y_revised))


