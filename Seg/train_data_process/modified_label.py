# -*- encoding: utf-8 -*-
"""
batch modified your label
json_dir : your json label save dir

"""
import os
import json


json_dir = r''
json_files = os.listdir(json_dir)


for json_file in json_files:
    if json_file.endswith('.png'):
        continue
    jsonfile = json_dir +'/'+ json_file
    # read your one json
    with open(jsonfile,'r',encoding = 'utf-8') as jf:

        info = json.load(jf)
        # print(type(info))
        # find 'a' replace to 'b'
    for i,label in enumerate(info['shapes']):
        if info['shapes'][i]['label'] == 'a':
                info['shapes'][i]['label'] = 'b'

    print(jsonfile)

    with open(jsonfile,'w') as new_jf:
        json.dump(info,new_jf)       

print('change name over!')