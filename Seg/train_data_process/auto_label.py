# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import json
from glob import glob
import copy
import matplotlib.pyplot as plt
# -----------------------------------------
from torchvision import transforms
import torch
from time import time
from PIL import Image
import segmentation_models_pytorch as smp


# 缺陷类别
defect_class = ["background", "super", "incomplete", "hopping", "streaking", "lattice", "ripple"]
color_dict = {'super': (255, 0, 255), 'incomplete': (0, 255, 0), 'hopping': (0, 0, 255), 'streaking': (255, 0, 0),
              'ripple': (240, 32, 160), 'lattice': (47, 255, 173), 'background': (150, 150, 150)
              }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_json_label(json_path):
    with open(json_path, "r") as fr:
        label_info = json.load(fr)
    img_path = os.path.join(os.path.split(json_path)[0], label_info['imagePath'])
    img = cv2.imread(img_path)
    out_pic = cv2.resize(img, (900, 900), interpolation=cv2.INTER_AREA)
    shapes = label_info["shapes"]
    for shape in shapes:
        category = shape["label"]
        points = shape["points"]
        mark_type = shape["shape_type"]
        if mark_type == 'circle':
            # 圆心坐标
            center_x, center_y = np.int32(points[0][0]), np.int32(points[0][1])
            point_x, point_y = np.int32(points[1][0]), np.int32(points[1][1])
            # 圆的半径
            circle_r = np.sqrt(np.power(abs(point_x - center_x), 2) + np.power(abs(point_y - center_y), 2))
            cv2.circle(img, (center_x, center_y), int(circle_r), color_dict[category], -1)
        else:
            # 封闭区域填充
            points_array = np.array(points, dtype=np.int32)
            cv2.fillPoly(img, [points_array], color_dict[category])
    out_mask = cv2.resize(img, (900, 900), interpolation=cv2.INTER_AREA)
    cmp_img = np.hstack((out_pic, out_mask))  # 对比标签
    cv2.imshow(label_info['imagePath'], cmp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ------------------------------  check json label  ----------------------------------
# for item in glob('./train_ok/*.json'):
#     check_json_label(item)
# for num in range(406, 432):
#     item = './with_all_label/%i.json' % num
#     check_json_label(item)
# check_json_label('F:/tmp_pic/7-PF.json')
# check_json_label('./with_all_label/306.json')


def check_mask(img_path, mask_path, save_img=False):
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    # mask = cv2.imread(mask_path, flags=0)
    mask = np.load(mask_path)
    orgin_img = copy.deepcopy(img)
    out_img = cv2.resize(img, (840, 840), interpolation=cv2.INTER_CUBIC)
    # out_mask = cv2.resize(mask, (840, 840))
    # 翘曲
    img[mask[:, :] == 1, 0] = 255   # B
    img[mask[:, :] == 1, 1] = 0     # G
    img[mask[:, :] == 1, 2] = 255   # R
    # 铺粉不全
    img[mask[:, :] == 2, 0] = 0     # B
    img[mask[:, :] == 2, 1] = 255   # G
    img[mask[:, :] == 2, 2] = 0     # B
    # 振纹
    img[mask[:, :] == 3, 0] = 0     # B
    img[mask[:, :] == 3, 1] = 0     # G
    img[mask[:, :] == 3, 2] = 255   # R
    # 拖拽
    img[mask[:, :] == 4, 0] = 255   # B
    img[mask[:, :] == 4, 1] = 0     # G
    img[mask[:, :] == 4, 2] = 0     # R
    # 点阵支撑
    img[mask[:, :] == 5, 0] = 47  # B
    img[mask[:, :] == 5, 1] = 255  # G
    img[mask[:, :] == 5, 2] = 173  # R
    # 水波纹
    img[mask[:, :] == 6, 0] = 240   # B
    img[mask[:, :] == 6, 1] = 32    # G
    img[mask[:, :] == 6, 2] = 160   # R
    img_label = cv2.resize(img, (840, 840), interpolation=cv2.INTER_CUBIC)
    cmp_img = np.hstack((out_img, img_label))  # 对比标签
    if save_img:
        cv2.imwrite('cmp' + img_name, cmp_img)
    cv2.imshow(img_name, cmp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


pic_list = glob('./data/images/*.png')
for num in range(1, len(pic_list)+1):
# # for num in range(406, 432):
    img_path = os.path.join('./data/images', str(num) + '.png')
    mask_path = os.path.join('./data/masks', str(num) + '.npy')
    check_mask(img_path, mask_path)


# for item in glob('./data/images/*.png'):
#     img_dir, img_name = os.path.split(item)
#     mask_dir = img_dir.replace('images', 'masks')
#     mask_path = os.path.join(mask_dir, img_name.replace('.png', '.npy'))
#     # mask_path = os.path.join(mask_dir, img_name)
#     check_mask(item, mask_path)
# check_mask('./data/images/353.png', './data/masks/353.npy')


# net = smp.UnetPlusPlus(encoder_name="resnet18", in_channels=1, classes=6)
# net.load_state_dict(torch.load('./tmp/train_epoch_800_avg_iou_0.963551.pth'))
# net.eval()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if device.type == 'cuda':
#     net.to(device)  # model move to GPU
# print(' --- model load complete --- ')
net = smp.UnetPlusPlus(encoder_name="resnet18", in_channels=1, classes=6)
net.load_state_dict(torch.load('./train_epoch_800_avg_iou_0.963551.pth'))
net.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    net.to(device)  # model move to GPU
def dep_label_super(img_path, append_bf=True, show_img=False, spec_class=None):
    # net = smp.UnetPlusPlus(encoder_name="resnet18", in_channels=1, classes=7)
    # net.load_state_dict(torch.load('./show_flask/tmp/hp_epoch_74_loss_0.000650.pth'))
    # net.load_state_dict(torch.load('./show_flask/tmp/epoch_116_res18_loss_0.011293.pth'))
    img_orgin = Image.open(img_path).convert('L')  # 图像转单通道
    img_h, img_w = img_orgin.size
    json_path = os.path.splitext(img_path)[0] + '.json'
    exist_json = os.path.exists(json_path)
    img_name = os.path.split(img_path)[1]
    if exist_json and append_bf:
        with open(json_path, 'r') as fr:
            json_info = json.load(fr)
    else:
        json_info = {"version": "5.0.01",
                     "flags": {},
                     "shapes": [],
                     "imagePath": img_name,
                     "imageData": None,
                     "imageHeight": img_h,
                     "imageWidth": img_w
                     }
    img = img_orgin.resize((1024, 1024), Image.LANCZOS)  # 图像缩放1024 * 1024
    out_orgin = cv2.cvtColor(cv2.resize(np.asarray(img_orgin), (1024, 1024), interpolation=cv2.INTER_CUBIC),
                             cv2.COLOR_GRAY2BGR)
    img_tensor = transforms.ToTensor()(img)  # 图像转张量
    input_tensor = img_tensor.unsqueeze(0)
    if device.type == 'cuda':
        input_tensor = input_tensor.to(device)
    with torch.no_grad():
        start_time = time()
        detect_result = net(input_tensor)
        end_time = time()
        print('detect image cost time: %0.4f' % (end_time - start_time))
        out_img = torch.argmax(detect_result, dim=1)
        out_array = np.uint8(out_img.squeeze().cpu().numpy())  # 掩膜
        out_pic = cv2.resize(out_array, (img_h, img_w), interpolation=cv2.INTER_CUBIC)  # 掩膜放大至原始大小
        # 使用深度学习标记翘曲
        for abnormal_class in defect_class[1:]:
            # if abnormal_class == spec_class:
                detect_index = defect_class.index(abnormal_class)
                info_binary = np.uint8((out_pic[:, :] == detect_index) * 1)
                contours, _ = cv2.findContours(info_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # contours, _ = cv2.findContours(info_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
                # cv2.drawContours(out_orgin, contours, -1, color_dict[abnormal_class], 1)
                for points in contours:
                    label_info = {'label': abnormal_class, 'points': [], 'group_id': '0', 'shape_type': 'polygon',
                                  'flags': {}}
                    if len(points) > 1:
                        list_value = points.squeeze()
                        # label_info['points'] = [[img_h * point[0] / 1024, img_w * point[1] / 1024] for point in list_value]
                        label_info['points'] = [[int(point[0]), int(point[1])] for point in list_value]
                        json_info['shapes'].append(label_info)
                    else:
                        # label_info['points'] = [[img_h * points.squeeze()[0] / 1024, img_w * points.squeeze()[1] / 1024]]
                        label_info['points'] = [[int(points.squeeze()[0]), int(points.squeeze()[1])]]
                        json_info['shapes'].append(label_info)
                # json文件重写
                with open(json_path, 'w') as fw:
                    json.dump(json_info, fw)
                # ------------------------------ modify json file end ----------------------------------
                print(json_path, ' : - - - - - - down - - - - - -')
                if show_img:
                    cv2.imshow("img", out_orgin)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            # else:
            #     continue

# check_mask('./data/images/751.png', './data/masks/751.png')

# for item in glob('./data/images/' + '*.png'):
#     mask_path = item.replace('images', 'masks')
#     check_mask(item, mask_path)
# ------------------------------  label super  ----------------------------------
for item in glob('./data/images/' + '*.png'):
    dep_label_super(item, append_bf=True, show_img=False)
# dep_label_super('./data/images/3.png', append_bf=False, spec_class='incomplete')




