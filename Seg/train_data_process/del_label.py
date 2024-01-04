# -*- encoding: utf-8 -*-

import os
import json
""" 
batch delete some class in your json file
json dir : json path 

"""

json_dir = r'D:\cjj\data\train_data\seg\labelme\9.18_all_copy\confused'
json_files = os.listdir(json_dir)

json_dict = {}


for json_file in json_files:
    if json_file.endswith('.png'):
        continue
    jsonfile = json_dir +'/'+ json_file
    
    with open(jsonfile,'r',encoding = 'utf-8') as jf:

        info = json.load(jf)

    for item in info['shapes']:
        if item['label'] == 'impression':
            info['shapes'].remove(item)
            print('del impression')
        elif item['label'] == 'unknow':
            info['shapes'].remove(item)
        json_dict = info
    print(jsonfile)

    with open(jsonfile,'w') as new_jf:
        json.dump(info,new_jf)       

print('del name over!')