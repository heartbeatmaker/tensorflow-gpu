'''이미지 확장자를 jpg로 바꾸는 코드'''

import os
import cv2
from PIL import Image
import shutil

input_path = "./image_downloaded/pikachu/"
image_paths = []


if os.path.isdir(input_path):
    # os.listdir : 이 디렉토리에 있는 전체 파일의 이름을 리스트 형태로 반환한다
    for inp_file in os.listdir(input_path):
        image_paths += [input_path + inp_file]

for image_path in image_paths:

    filename = image_path.split('/')[-1]
    extension = filename.split('.')[-1]

    filename_new = filename.split('.')[0]+".jpg"

    if extension != "jpg":
        if extension == "png" or extension == "jpeg":
            os.rename(image_path, input_path+filename_new)
            print("Changed filename: "+filename+" -> "+filename_new)
        else:
            os.remove(image_path)
            print("Removed "+filename)






