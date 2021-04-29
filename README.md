# Keras Faster-RCNN for TensorFlow 2.3 

## Update

The applications is upgraded to support both TensorFlow 2.3 keras 2.4. If you have any issues, please keep in touch. 

## Requirements

The scripts are upgraded to support Python3.7. It is very necessary for users to intall compatible opencv-python and 
opencv-contrib-python versions in order to prevent QT errors. The following versions are recommended to be installed. 

* TensorFlow 2.3
* Keras 2.4
* opencv-python==4.1.1.26
* opencv-contrib-python==4.1.1.26

## Out of box model to predict

The original author has trained a model to predict kitti.  

## Kitti Dataset:

kitti website: 

http://www.cvlibs.net/datasets/kitti/eval_object.php

data_object_label_2

$ wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

data_object_image_2

$ wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip

## Train the model based on ResNet50

1st. Create the fileholders including data, model, image, result_images respectively.  

     Please change user to your linux name 

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

2nd. Test a specific image 
python test_frcnn_kitti.py -p ./images/00009.png

3rd. Test all images in a designated fileholder 
python test_frcnn_kitti.py -p ./images
```
You can also using `-p` to specific single image to predict, or send a path contains many images, the application will automatically recognise the images.

## Results

It shows the following results during running the train script. While run the script, it saves time to set both epocks and lengh as 10. 
After a success run, users can set them to a large number to optimize the tainning effect. 

1st.Parsing annotation files

    Config has been written to config.pickle, and can be loaded when testing to ensure correct results

    Training images per class:
    {'Car': 28742,
     'Cyclist': 1627,
     'DontCare': 11295,
     'Misc': 973,
     'Pedestrian': 4487,
     'Person_sitting': 222,
     'Tram': 511,
     'Truck': 1094,
     'Van': 2914,
     'bg': 0}
     
    Num classes (including bg) = 10
    Num train samples 6276
    Num val samples 1205

........

2nd. Starting training: 

    Epoch 1/10
    2021-04-28 14:08
    1/10 [==>...........................] - ETA: 3:34 - rpn_cls: 5.1244 - rpn_regr: 0.9318  
    2/10 [=====>........................] - ETA: 2:34 - rpn_cls: 5.0801 - rpn_regr: 0.9644 - detector_cls: 2.3001 - detector_regr: 0.3942
    3/10 [========>.....................] - ETA: 2:10 - rpn_cls: 5.0417 - rpn_regr: 0.9217
    ........

    Epoch 8/10
    Average number of overlapping bounding boxes from RPN = 2.0 for 10 previous iterations
    1/10 [==>...........................] - ETA: 33s - rpn_cls: 7.6447 - rpn_regr: 0.2620  
    2/10 [=====>........................] - ETA: 27s - rpn_cls: 6.3826 - rpn_regr: 0.3558 
    3/10 [========>.....................] - ETA: 21s - rpn_cls: 6.4132 - rpn_regr: 0.3519 - detector_cls: 0.2926 - detector_regr: 0.4456
    ........

    Average number of overlapping bounding boxes from RPN = 1.6 for 10 previous iterations
    10/10 [==============================] - 94s 9s/step - rpn_cls: 4.8054 - rpn_regr: 0.5197 - detector_cls: 0.3994 - detector_regr: 0.5194
    Mean number of bounding boxes from RPN overlapping ground truth boxes: 1.391304347826087
    Classifier accuracy for bounding boxes from RPN: 0.9
    Loss RPN classifier: 4.342926752567291
    Loss RPN regression: 0.42844736129045485
    Loss Detector classifier: 0.45890533179044724
    Loss Detector regression: 0.4877036601305008
    Elapsed time: 94.26770877838135

## Issues 

1st. WARNING:tensorflow:

    5 out of the last 6 calls to <function Model.make_train_function.<locals>.train_function at 0x7f8db04f8170> triggered tf.function retracing.  
    Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with
    different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), 
    @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to
    https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for
    more details.

2nd. Floating incompatibility issue

    There is a floating incompatibility issue during the training. It might be an issue related to the new scripts related to TensorFlow 2.3 and Keras 
    2.4. But it is not sure. 

    TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int64 of argument 'x'.

**That's all, help you enjoy!**
