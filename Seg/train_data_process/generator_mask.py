# -*- coding: utf-8 -*-
import os
import cv2
import json
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


"""
json transform mask script 
defect_dict : a dict of defect ex: 'cat':1
file_dir : json and image path
save_path: save mask and images
target_size : your data resize to crop size
"""
defect_dict = { } 


def json_to_mask(file_dir,save_path, target_size=(1024, 1024), start_num=None):

    a = set()
    pic_list = glob(file_dir + '/*.png')
    end_num = len(pic_list)
    if start_num:
        pic_list = [os.path.join(file_dir, '%i.png' % num) for num in range(start_num, end_num)]
    tar_h, tar_w = target_size
    for item in pic_list:
        defect_list = []
        _, img_name = os.path.split(item)
        json_file = item.replace('.png', '.json')

        img = cv2.imread(item, flags=0)
        img_h, img_w = img.shape

        mask = np.zeros((tar_h, tar_w), np.uint8)
        with open(json_file, "r") as f:
            label = json.load(f)
        shapes = label["shapes"]

        for shape in shapes:
            category = shape["label"]
            # if category in ('ripple', 'incomplete'):
            #     continue
            defect_list.append(category)
            mark_type = shape["shape_type"]
            a.add(mark_type)
            points = [[tar_w * point[0] / img_w, tar_h * point[1] / img_h] for point in shape["points"]]
            if mark_type == 'circle':
                
                center_x, center_y = np.int32(points[0][0]), np.int32(points[0][1])
                point_x, point_y = np.int32(points[1][0]), np.int32(points[1][1])
                
                circle_r = np.sqrt(np.power(abs(point_x - center_x), 2) + np.power(abs(point_y - center_y), 2))
                cv2.circle(mask, (center_x, center_y), int(circle_r), defect_dict[category], -1)
            else:
                
                points_array = np.int32(np.around(points))
                cv2.fillPoly(mask, [points_array], defect_dict[category])
        print(a)
        matrix_set = set(np.unique(mask).tolist())
        defect_list.append('background')
        label_set = {defect_dict[item] for item in set(defect_list)}
        if matrix_set == label_set:
            print('------ all label ok ------')
        else:
            print('------ label failed ------')
        if img_h > target_size[0]:
            out_pic = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)  # resize to smaller  INTER_AREA
        else:
            out_pic = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)  # resize to biger  INTER_CUBIC
        image_path = os.path.join(save_path,'images')
        mask_path = os.path.join(mask_path,'masks')
        
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        
        if not os.path.exists( mask_path):
            os.makedirs(mask_path)
        cv2.imwrite(image_path + "/%s" % img_name, out_pic, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_path + "/%s" % img_name, mask)

        print(item, ' : --- down ---')


if __name__ == '__main__': 
    file_dir = ""  
    save_path = ""
    json_to_mask(file_dir,save_path)
