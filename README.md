# Keras Faster-RCNN

**[UPDATE]**

This work has been publiced on **StrangeAI - An AI Algorithm Hub**,  You can found this work at [Here](http://ai.loliloli.pro/) (You may found more interesting work on this website, it's a very good resource to learn AI, StrangeAi authors maintainered all applications in AI).

## Update

This code is upgraded to support to both TensorFlow 2.3 keras 2.4. If you have any issues, please keep in touch. 

## Requirements

Basically, this code supports both python2.7 and python3.7, the following package should be installed:

* TensorFlow 2.3
* Keras 2.4
* opencv-python==4.1.1.26
* opencv-contrib-python==4.1.1.26

## Out of box model to predict

The original author has trained a model to predict kitti. I will update a dropbox link here later. 

## Kitti Dataset:

kitti website: 

http://www.cvlibs.net/datasets/kitti/eval_object.php

data_object_label_2

$ wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

data_object_image_2

$ wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip

## Train the model based on ResNet50

1st. Create the fileholders including data, model, image, result_images respectively.  

     $ mkdir /home/user/Documents/keras_frcnn/data
     ./Documents/keras_frcnn/data
     
     Take the same way to create other folders. 
     ./Documents/keras_frcnn/images 
     ./Documents/keras_frcnn/result_images
     ./Documents/keras_frcnn/model 
     
2nd. Generate the file of kitti_simple_label.txt

     Enter into the fileholder of keras_frcnn
     $ cd ./Documents/keras_frcnn
     
     Run the kitti script 
     $ python generate_simple_kitti_anno_file.py \
     /home/user/Documents/keras_frcnn/data/training/image_2 \
     /home/user/Documents/keras_frcnn/data/training/label_2

3nd. set up both resnet50 and kitti_frcnn weight

     A.Set ResNet50 in config.py under the second directory of keras_frcnn. 
     
     self.network = 'resnet50'
     self.model_path = './model/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
     
     B.Change the original vgg to kitti_frcnn.last.hdf5 in config.py
     
     self.model_path = '.model/kitti_frcnn.last.hdf5'
     
     C.Set base_net_weights in train_frcnn_kitti.py. 
     
     try:
     print('loading weights from {}'.format(cfg.base_net_weights))
     # -model_rpn.load_weights(cfg.model_path, by_name=True)
     model_rpn.load_weights(cfg.base_net_weights, by_name=True)
     # -model_classifier.load_weights(cfg.model_path, by_name=True)
     model_classifier.load_weights(cfg.base_net_weights, by_name=True)
     
     D. Set restnet50 weight
     
     def get_weight_path():
     if K.image_data_format() == 'channels_first':
        return 'resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
     else:
        return 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
     
4th. Train the model

     Enter into the fileholder of keras_frcnn
     $ cd ./Documents/keras_frcnn
     
     Run the train script 
     $ python train_frcnn_kitti.py
     

## Train New Dataset

To train a new dataset is also very simple and straight forward. Simply convert your detection label file whatever format into this format:

```
/path/training/image_2/000000.png,712.40,143.00,810.73,307.92,Pedestrian
/path/training/image_2/000001.png,599.41,156.40,629.75,189.25,Truck
```
Which is `/path/to/img.png,x1,y1,x2,y2,class_name`, with this simple file, we don't need class map file, our training program will statistic this automatically.

## For Test 

If you want see how good your trained model is, please simply run the script as follows. 
```
$ cd ./Documents/keras_frcnn

1st. Test the default image
$ python test_frcnn_kitti.py

2nd. Testa a specific image 
python test_frcnn_kitti.py -p ./images/00009.png

3rd. Test all images in a designated fileholder 
python test_frcnn_kitti.py -p ./images
```
You can also using `-p` to specific single image to predict, or send a path contains many images, our program will automatically recognise that.

**That's all, help you enjoy!**
