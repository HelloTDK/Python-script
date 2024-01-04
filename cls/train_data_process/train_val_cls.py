
"""random shuffle classification data to a new_dir,split your data to train and val"""
import os
import shutil
import random
"""
data_dir : your cls data source
new_dir : your save dir 
"""
data_dir = r""
new_dir = r""

powders = os.listdir(os.path.join(data_dir,"powder"))
others  = os.listdir(os.path.join(data_dir,"others"))
if not os.path.exists(new_dir):
    os.mkdir(new_dir)
train_ratio = 0.8
nums_train_powder = int(train_ratio*len(powders))
# nums_val_powder = int(len(powders) - nums_train_powder)
nums_train_others = int(train_ratio*len(others))
# nums_val_others = int(len(others) - nums_train_others)
train_powder = powders[:nums_train_powder]
random.shuffle(train_powder)
print(len(train_powder))
val_powder = powders[nums_train_powder:]
random.shuffle(val_powder)
print(len(val_powder))
train_others = others[:nums_train_others]
random.shuffle(train_others)
print(len(train_others))
val_others = others[nums_train_others:]
random.shuffle(val_others)
print(len(val_others))
if not os.path.exists(os.path.join(new_dir,'train','powder')):
    os.makedirs(os.path.join(new_dir,'train','powder'))
if not os.path.exists(os.path.join(new_dir,'val','powder')):
    os.makedirs(os.path.join(new_dir,'val','powder'))
if not os.path.exists(os.path.join(new_dir,'train','others')):
    os.makedirs(os.path.join(new_dir,'train','others'))
if not os.path.exists(os.path.join(new_dir,'val','others')):
    os.makedirs(os.path.join(new_dir,'val','others'))
for item in train_powder:
    shutil.copyfile(os.path.join(data_dir,'powder',item),os.path.join(new_dir,'train','powder',item))
for item in val_powder:
    shutil.copyfile(os.path.join(data_dir,'powder',item),os.path.join(new_dir,'val','powder',item))
for item in train_others:
    shutil.copyfile(os.path.join(data_dir,'others',item),os.path.join(new_dir,'train','others',item))
for item in val_others:
    shutil.copyfile(os.path.join(data_dir,'others',item),os.path.join(new_dir,'val','others',item))