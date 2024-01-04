import os
import cv2
import numpy as np
from PIL import Image
import json
def save_json(npy_path,save_json_path):
    if not os.path.exists(save_json_path):
        os.mkdir(save_json_path)
    npys = os.listdir(npy_path)
    for npy in npys:
        print(npy)


        if npy.split('.')[1] == 'npy':
            out_origin = np.load(os.path.join(npy_path, npy))
            img_h, img_w = out_origin.shape
        else:
            out_origin =cv2.imread(os.path.join(npy_path, npy),0)
            img_h, img_w = out_origin.shape
        img_name = npy.split('.')[0] + '.png'
        json_name = npy.split('.')[0] + '.json'
        json_info = {"version": "5.0.01",
                     "flags": {},
                     "shapes": [],
                     "imagePath": img_name,
                     "imageData": None,
                     "imageHeight": img_h,
                     "imageWidth": img_w
                     }
        # with open(save_json_path + '/' + npy, 'r') as f:
        #     coco_output = json.load(f)
        # img = npy.replace('.npy','.png')

        json_info =func(out_origin,json_info)
        with open(os.path.join(save_json_path,json_name), 'w') as fw:
            json.dump(json_info, fw)
        a = 1
def func(out_origin,json_info):
    out_info = []
    h, w = out_origin.shape
    # out_orgin = np.zeros((h,w),np.uint8)
    super_binary = np.uint8((out_origin[:, :] == 1) * 1)
    contours, _ = cv2.findContours(super_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for points in contours:
        label_info = {'label': 'super', 'points': [], 'group_id': '0', 'shape_type': 'polygon','flags': {}}
        if len(points) > 1:
            label_info['points'] = [[int(point[0]), int(point[1])] for point in points.squeeze()]
            json_info['shapes'].append(label_info)
        else:
            label_info['points'] = [[int(points.squeeze()[0]), int(points.squeeze()[1])]]
            json_info['shapes'].append(label_info)
    incomplete_binary = np.uint8((out_origin[:, :] == 2) * 1)
    contours, _ = cv2.findContours(incomplete_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for points in contours:
        label_info = {'label': 'incomplete', 'points': [], 'group_id': '1', 'shape_type': 'polygon','flags': {}}
        if len(points) > 1:
            label_info['points'] = [[int(point[0]), int(point[1])] for point in points.squeeze()]
            json_info['shapes'].append(label_info)
        else:
            label_info['points'] = [[int(points.squeeze()[0]), int(points.squeeze()[1])]]
            json_info['shapes'].append(label_info)
    hopping_binary = np.uint8((out_origin[:, :] == 3) * 1)
    contours, _ = cv2.findContours(hopping_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for points in contours:
        label_info = {'label': 'hopping', 'points': [], 'group_id': '2', 'shape_type': 'polygon','flags': {}}
        if len(points) > 1:
            label_info['points'] = [[int(point[0]), int(point[1])] for point in points.squeeze()]
            json_info['shapes'].append(label_info)
        else:
            label_info['points'] = [[int(points.squeeze()[0]), int(points.squeeze()[1])]]
            json_info['shapes'].append(label_info)
    
    streaking_binary = np.uint8((out_origin[:, :] == 4) * 1)
    contours, _ = cv2.findContours(streaking_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for points in contours:
        label_info = {'label': 'streaking', 'points': [], 'group_id': '2', 'shape_type': 'polygon','flags': {}}
        if len(points) > 1:
            label_info['points'] = [[int(point[0]), int(point[1])] for point in points.squeeze()]
            json_info['shapes'].append(label_info)
        else:
            label_info['points'] = [[int(points.squeeze()[0]), int(points.squeeze()[1])]]
            json_info['shapes'].append(label_info)
    lattice_binary = np.uint8((out_origin[:, :] == 5) * 1)
    contours, _ = cv2.findContours(lattice_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for points in contours:
        label_info = {'label': 'lattice', 'points': [], 'group_id': '2', 'shape_type': 'polygon','flags': {}}
        if len(points) > 1:
            label_info['points'] = [[int(point[0]), int(point[1])] for point in points.squeeze()]
            json_info['shapes'].append(label_info)
        else:
            label_info['points'] = [[int(points.squeeze()[0]), int(points.squeeze()[1])]]
            json_info['shapes'].append(label_info)
    ripple_binary = np.uint8((out_origin[:, :] == 6) * 1)
    contours, _ = cv2.findContours(ripple_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for points in contours:
        label_info = {'label': 'ripple', 'points': [], 'group_id': '2', 'shape_type': 'polygon','flags': {}}
        if len(points) > 1:
            label_info['points'] = [[int(point[0]), int(point[1])] for point in points.squeeze()]
            json_info['shapes'].append(label_info)
        else:
            label_info['points'] = [[int(points.squeeze()[0]), int(points.squeeze()[1])]]
            json_info['shapes'].append(label_info)
    out_info = json_info
    return out_info

if __name__ == '__main__':
    npy_path =r'D:\cjj\PF_seg\algorithm\mmsegmentation\maskformer_all.pkl'
    save_json_path = r'D:\cjj\PF_seg\algorithm\mmsegmentation\maskformer_all.pkl_json'
    save_json(npy_path, save_json_path)





