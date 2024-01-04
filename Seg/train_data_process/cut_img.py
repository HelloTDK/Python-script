import cv2
import os
"""
cut one image to four images and save four images 
img_path : your image path 
save_path : save four images path
"""
def batch_cut_img(img_path,save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_names = os.listdir(img_path)
    for img_name in img_names:
        if img_name.endswith('png'):
            path = os.path.join(img_path,img_name)
            im = cv2.imread(path,0)
            h,w = im.shape[:2]
            half_h = round(h//2)
            half_w  = round(w//2)
            im0 = im[0:half_h,0:half_w]
            im1 = im[0:half_h,half_w:w]
            im2 = im[half_h:h,0:half_w]
            im3 = im[half_h:h,half_w:w]
            im0_name = img_name.split('.')[0]+'_0.png'
            im1_name = img_name.split('.')[0]+'_1.png'
            im2_name = img_name.split('.')[0]+'_2.png'
            im3_name = img_name.split('.')[0]+'_3.png'
            cv2.imwrite(os.path.join(save_path,im0_name),im0)
            cv2.imwrite(os.path.join(save_path,im1_name),im1)
            cv2.imwrite(os.path.join(save_path,im2_name),im2)
            cv2.imwrite(os.path.join(save_path,im3_name),im3)
if __name__ == "__main__":
    img_path = r''
    save_path = r''
    batch_cut_img(img_path,save_path)