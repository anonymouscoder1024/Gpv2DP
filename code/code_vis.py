import numpy
import os,shutil
import binascii
from random import *
import color_map
import cv2
import time

rs = Random()

img_path = '../data/img/'
color_path = '/gray_img'


def getMatrixfrom_bin(filename, width = 512, oneRow = False):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)
    fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])
    if oneRow is False:
        rn = len(fh)/width
        fh = numpy.reshape(fh[:rn*width],(-1,width))
    fh = numpy.uint8(fh)
    return fh


def getMatrixfrom_asm(filename, startindex = 0, pixnum = 89478485):
    with open(filename, 'rb') as f:
        f.seek(startindex, 0)
        content = f.read(pixnum)
    hexst = binascii.hexlify(content)
    fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])
    fh = numpy.uint8(fh)
    return fh


def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)
    size = fsize/float(1024)
    return round(size,2)


if __name__ == '__main__':
    for txtfile in os.listdir('../data//txt'):
        if txtfile == '.DS_Store':
            continue
        if txtfile == '._.DS_Store':
            continue
        if txtfile == 'readme.md':
            continue

        # images save path, distinguish buggy and clean
        project_name = txtfile.split('.txt')[0]
        path_img = img_path + project_name + color_path

        if not os.path.exists(path_img):
            os.makedirs(path_img)

        if not os.path.exists(path_img + '/buggy/'):
            os.makedirs(path_img + '/buggy/')

        if not os.path.exists(path_img + '/clean/'):
            os.makedirs(path_img + '/clean/')

        filename = '../data/txt/'+txtfile
        f = open(filename)
        num = 0
        no_num = 0
        img_paths_and_labels = []
        for line in f:

            # load file path
            f_path = '../data/archives/' + project_name + '/' + line[:-3]
            f_path = line[:-3]
            label = line[-2:-1]
            start = time.perf_counter()
            if os.path.exists(f_path):
                num = num + 1
                size = get_FileSize(f_path)
                if size == 0:
                    break
                im = color_map.get_new_color_img(f_path)
                buggy_or_clean = '/clean/_'
                if label == '1':
                    buggy_or_clean = '/buggy/_'

                # generate images
                path_save = path_img+buggy_or_clean+''.join(line[:-3]).replace('/', '_')+'.png'
                path_save = path_save.replace('.java', '')
                cv2.imwrite(path_save, im)

                # save images
                img_paths_and_labels.append([path_save, label])

            else:
                no_num = no_num + 1
            end = time.perf_counter()
            image_time = str(end-start)

        # save images
        numpy.savetxt(path_img + '/instances.txt', img_paths_and_labels, fmt="%s", delimiter=" ")

        print(project_name+" num:"+str(num))
        print(project_name+" no num:"+str(no_num))
