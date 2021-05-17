#!/usr/bin/env python
# coding: utf-8

"""

Parser for the data concistency 

1.Path of kitti_simple_label

./Documents/keras_frcnn/data/training/image_2/002960.png,393.84,175.87,496.36,249.54,Car
It will be arranged in the formet of (filename, x1, y1, x2, y2, class_name)

2.all_img_data—>all_imgs

The file of all_imgs is a list with each image message saved in the dictionary. Each element 
of all_imgs include the content of ['filepath','width','height','imageid','imageset','bbox'], 
and it is in the annotation_data with the xml format. 

{
'filepath': '/home/mic/Documents/keras_frcnn/VOCdevkit/VOC2012/JPEGImages/2012_003845.jpg', 
'width': 333, 
'height': 500, 
'bboxes': [{
            'class': 'person', 
            'x1': 29, 
            'x2': 163, 
            'y1': 108, 
            'y2': 406, 
            'difficult': False
          }], 
'imageset': 'trainval'
}

3.class_mapping

class_mapping：
{'person': 0, 'dog': 1, 'bottle': 2, 'motorbike': 3, 'train': 4, 'bus': 5, 'car': 6, 'bird': 7, 
 'aeroplane': 8, 'sofa': 9, 'horse': 10, 'cat': 11, 'cow': 12, 'chair': 13, 'pottedplant': 14, 
 'sheep': 15, 'diningtable': 16, 'bicycle': 17, 'boat': 18, 'tvmonitor': 19, 'bg': 20}

For any dictonary with the key-value not being the data form of 'bg': 20 , need to switch the 
the items, for instance： 'motorbike': 20——>'motorbike': 20, 'bg': 3——>'bg':20

{'person': 0, 'dog': 1, 'bottle': 2, 'motorbike': 20, 'train': 4, 'bus': 5, 'car': 6, 'bird': 7, 
 'aeroplane': 8, 'sofa': 9, 'horse': 10, 'cat': 11, 'cow': 12, 'chair': 13, 'pottedplant': 14, 
 'sheep': 15, 'diningtable': 16, 'bicycle': 17, 'boat': 18, 'tvmonitor': 19, 'bg': 3}

4.classes_count：

{'person': 24252, 'dog': 1598, 'bottle': 1561, 'motorbike': 801, 'train': 704, 'bus': 685, 
 'car': 2492, 'bird': 1271, 'aeroplane': 1002, 'sofa': 841, 'horse': 803, 'cat': 1277, 'cow': 771, 
 'chair': 3056, 'pottedplant': 1202, 'sheep': 1084, 'diningtable': 801, 'bicycle': 837, 
 'boat': 1059, 'tvmonitor': 893, 'bg': 0}

5.Exchange between list comprehension and for statments 

1).List comprehension：
key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
print(key_to_switch)
# print result
# motorbike

2).for statements：
if class_mapping['bg'] != len(class_mapping) - 1:
    for key in class_mapping.keys():
        if class_mapping[key] == len(class_mapping) - 1:
            key_to_switch = key
            print(key_to_switch)
# print result 
# motorbike
"""


import cv2
import numpy as np


def get_data(input_path):
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    visualise = True

    with open(input_path, 'r') as f:
        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split # filename is a list 

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and not found_bg:
                    print('Found class name with special name bg. Will be treated as a'
                          ' background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping) # class_mapping is the label of each class

            if filename not in all_imgs:
                all_imgs[filename] = {}
                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval' # s['imageset'] == 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'     # s['imageset'] == 'test']　

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)),
                 'y2': int(float(y2))})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # Make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping