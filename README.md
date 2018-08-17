# CRNN_Tensorflow

A Tensorflow implementation of a Deep Neural Network for scene text recognition mainly based on the paper "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition". You can refer to [their paper for details](http://arxiv.org/abs/1507.05717). Thanks to the author [Baoguang Shi](https://github.com/bgshih).  

This model consists of a CNN stage, an RNN stage and a CTC loss for the task of scene text recognition.

## Installation

There are Dockerfiles inside the folder `docker`. Follow the instructions in `docker/README.md` to build the images.

Other than that, this software has only been tested on ubuntu 16.04(x64), python3.5, cuda-8.0, cudnn-6.0 with a GTX-1070 GPU. To install this software you need tensorflow 1.3.0.
Other versions of tensorflow have not been tested but might work properly above version 1.0. Other required packages may be installed with

```
pip3 install -r requirements.txt
```

## Testng the pretrained model

This repository contains a model pretrained on a subset of the [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/) data set. A data preparation stage converts the dataset into tensorflow records which can be found in the data folder.

The pretrained model can be tested on the converted dataset with

```
python tools/test_shadownet.py --dataset_dir data/ --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999
```

The expected output is:

![Test output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_output.png)

In order to test a single image do:

```
python tools/demo_shadownet.py --image_path data/test_images/test_01.jpg --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999
```

`Example image_01 is`  
![Example image1](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/text_example_image1.png)  
`Expected output_01 is`  
![Example image1 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image1_output.png)  
`Example image_02 is`  
![Example image2](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image2.png)  
`Expected output_02 is`  
![Example image2 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image2_output.png) 
`Example image_03 is`  
![Example image3](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/demo_chinese.png)  
`Expected output_03 is`  
![Example image3 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/demo_chinese_output.png)
`Example image_04 is`  
![Example image4](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/dmeo_chinese_2.png)  
`Expected output_04 is`  
![Example image4 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/demo_chinese_2_ouput.png)

## Training the model

#### Data preparation

All image data must be stored in a root folder with two subfolders named "Train" and "Test" and one image per text snippet. An accompanying annotations file `sample.txt` is required for each of the subfolders. For each image this file details its path relative to the Train or Test directory and its corresponding text label. For example:

```
1/2/373_coley_14845.jpg coley
17/5/176_Nevadans_51437.jpg nevadans
```

These directories is then converted into tensorflow records with

```
python tools/write_text_features.py --dataset_dir path/to/your/dataset --save_dir path/to/tfrecords_dir
```

All training images will be scaled to (32, 100, 3). Images in `Train` will be divided into train and validation sets. The ratio can be changed with a parameter to `write_text_features.py` (call with `-h` to see the help).


#### Train model

For the pretrained model 40000 epochs where used, with a batch size of 32, initial learning rate of 0.1 decreased by a factor 0.1 every 10000 epochs. For more information on training parameters check the file `global_configuration/config.py`. Train the model with

```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords
```

You can also continue the training process from the snapshot by

```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords --weights_path path/to/your/last/checkpoint
```

After enough iteration the directory `logs` should show progress like this:
 
![Training log](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_log.png)

The seq distance is computed by calculating the distance between two sparse tensors so the lower the accuracy value is the better the model performs. The training accuracy is computed by calculating the character-wise precision between the prediction and the ground truth so the higher it is, the better the model performs.

During training of the provided model the loss dropped as follows

![Training loss](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_loss.png)

The distance between the ground truth and the prediction drops as follows  

![Sequence distance](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/seq_distance.png)

## Experiment

The accuracy during training process rises as follows  

![Training accuracy](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/training_accuracy.md)

## TODO

The model is trained on a subet of [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/). One could train again on the whold dataset for bette results since the CRNN model needs large amounts of training data.

