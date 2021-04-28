# Keras Faster-RCNN

**[UPDATE]**

This work has been publiced on **StrangeAI - An AI Algorithm Hub**,  You can found this work at [Here](http://ai.loliloli.pro/) (You may found more interesting work on this website, it's a very good resource to learn AI, StrangeAi authors maintainered all applications in AI).

## Update

This code only support to both TensorFlow 2.3 keras 2.4. If you can fix it, feel free to send me a PR.

## Requirements
Basically, this code supports both python2.7 and python3.7, the following package should be installed:

* TensorFlow 2.3
* Keras 2.4
* opencv-python==4.1.1.26
* opencv-contrib-python==4.1.1.26

## Out of box model to predict

I have trained a model to predict kitti. I will update a dropbox link here later. Let's see the result of predict:

## Train New Dataset

to train a new dataset is also very simple and straight forward. Simply convert your detection label file whatever format into this format:

kitti dataset downloads:

kitti website

http://www.cvlibs.net/datasets/kitti/eval_object.php

data_object_label_2
$ wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

data_object_image_2

$ wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip


```
/path/training/image_2/000000.png,712.40,143.00,810.73,307.92,Pedestrian
/path/training/image_2/000001.png,599.41,156.40,629.75,189.25,Truck
```
Which is `/path/to/img.png,x1,y1,x2,y2,class_name`, with this simple file, we don't need class map file, our training program will statistic this automatically.

## For Predict

If you want see how good your trained model is, simply run:
```
python test_frcnn_kitti.py
```
you can also using `-p` to specific single image to predict, or send a path contains many images, our program will automatically recognise that.

**That's all, help you enjoy!**
