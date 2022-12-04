# Keras Faster-RCNN for TensorFlow 2.4

## Update

The applications is upgraded to support both TensorFlow 2.4 and keras 2.4. If you have any issues, please keep in 
touch. Pllease have a look at the project_tree for understanding the strcture. 

## Requirements

The scripts are upgraded to support Python3.7. It is very necessary for users to intall compatible opencv-python and 
opencv-contrib-python versions in order to prevent QT errors. The following versions are recommended to be installed. 

* TensorFlow 2.4 at least 
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

## Train the model based on the pretrained ResNet50

1st. Create the fileholders including data, model, image, result_images respectively.  

     Please change user to your linux name 

     $ mkdir /home/user/Documents/keras_frcnn/data
     ./Documents/keras_frcnn/data
     
     Take the same way to create other folders. 
     ./Documents/keras_frcnn/images 
     ./Documents/keras_frcnn/result_images
     ./Documents/keras_frcnn/model 
     
2nd. Generate the file of kitti_simple_label.txt

     Please change user to your linux name 

     Enter into the fileholder of keras_frcnn
     $ cd ./Documents/keras_frcnn
     
     Run the kitti script 
     $ python generate_simple_kitti_anno_file.py \
     /home/user/Documents/keras_frcnn/data/training/image_2 \
     /home/user/Documents/keras_frcnn/data/training/label_2

3nd. Set up both resnet50 and kitti_frcnn weights

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
     
     D.Set the restnet50 weight
     
     def get_weight_path():
     if K.image_data_format() == 'channels_first':
        return 'resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
     else:
        return 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
     
4th. Train the new model

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

## Test the model 

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

## Operation Results

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
    1/10 [==>...........................] - ETA: 3:27 - rpn_cls: 9.5235 - rpn_regr: 0.8175  
    2/10 [=====>........................] - ETA: 2:26 - rpn_cls: 9.5661 - rpn_regr: 0.8448 - detector_cls: 2.2996 - detector_regr: 0.3951
    3/10 [========>.....................] - ETA: 1:41 - rpn_cls: 9.4287 - rpn_regr: 0.7807 - detector_cls: 2.2964 - detector_regr: 0.4211
    ........

    Epoch 8/10
    Average number of overlapping bounding boxes from RPN = 2.0 for 10 previous iterations
    1/10 [==>...........................] - ETA: 1:01 - rpn_cls: 1.9340 - rpn_regr: 0.1092  
    2/10 [=====>........................] - ETA: 42s - rpn_cls: 2.2123 - rpn_regr: 0.1378 - detector_cls: 0.4928 - detector_regr: 0.4933
    3/10 [========>.....................] - ETA: 43s - rpn_cls: 2.1131 - rpn_regr: 0.1400
    
    ........

    Average number of overlapping bounding boxes from RPN = 1.6 for 10 previous iterations
    8/10 [=======================>......] - ETA: 15s - rpn_cls: 1.8296 - rpn_regr: 0.2392 
    9/10 [==========================>...] - ETA: 7s - rpn_cls: 1.8296 - rpn_regr: 0.2419 - detector_cls: 0.4357 - detector_regr: 0.3452
    10/10 [==============================] - 79s 8s/step - rpn_cls: 1.8336 - rpn_regr: 0.2432 - detector_cls: 0.4441 - detector_regr: 0.3483
    Mean number of bounding boxes from RPN overlapping ground truth boxes: 2.55
    Classifier accuracy for bounding boxes from RPN: 0.840625
    Loss RPN classifier: 1.8696473240852356
    Loss RPN regression: 0.25528666824102403
    Loss Detector classifier: 0.5204675108194351
    Loss Detector regression: 0.3761039458215237
    Elapsed time: 79.11395478248596
    Training complete, exiting.


## Issues 

1st. WARNING:tensorflow:

    5 out of the last 6 calls to <function Model.make_train_function.<locals>.train_function at 0x7f8db04f8170> triggered tf.function retracing.  
    Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with
    different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), 
    @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to
    https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for
    more details.

2nd. Floating incompatibility issue

    There is a floating incompatibility issue during the training. It shows the error in losses.py  It is not useful to set dtype='float32' in either numpy or keras code in all avaiable
    scripts. So it is hard to make an improvement right now. 
    
    Exception: in user code:
    return step_function(self, iterator)
    /Documents/keras_frcnn/keras_frcnn/losses.py:59 class_loss_regr_fixed_num  *
        x = y_true[:, :, 4*num_classes:] - y_pred
    raise e
    TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int64 of argument 'x'.
    
    It is not useful to cast the expression with float32 as following. 
    x = K.cast((y_true[:, :, 4*num_classes:] - y_pred), 'float32')

3nd. Correct the code in data_generators.py 

    The runtime is improved greatly and the accuracy is also enhanced tremendously. 

    Delete the wrong line of code: 
    -val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
    Add the correct line of code:
    val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - (num_regions - num_pos))

**That's all, help you enjoy!**
