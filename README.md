TABLE OF CONTENT
---------------------

 * Introduction
 * Requirements
 * Preparing Dataset
 * Screenshots
 * Tech/Framework Used
 * Features
 * Usage
 * Credits
# Introduction

In this project, I am building a model that will learn to automatically analyze medical images for malaria testing.

## Requirements

First of all, you need to download the image dataset using link ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/cell_images.zip or go to [official NIH page](https://lhncbc.nlm.nih.gov/publication/pub9932) and download cell_images.zip file. After downloading extract the downloaded item. Make a new folder ```malaria``` and move the extracted folder in that folder.

Project directory should look like this-

![](images/directory_1.PNG)

Image dataset look like this-

![](images/our_dataset.jpg)

## Preparing Dataset

Since our dataset doesn't have pre split data for training, testing and validation, so we need to do it ourselvers.
To create our data split we are going to use ```build_dataset.py``` script.
  - Grab the paths to all our example images and randomly shuffle them.
  - Split the images paths into the training, validation, and testing.
  - Create three new sub-directories in the ```malaria/``` directory, namely ```training/``` , ```validation/``` , and ```testing/```.
  - Automatically copy the images into their corresponding directories.

After executing ```build_dataset.py``` project directory will lool similar to this-
![](images/directory_3.PNG)

## Tech/Framework Used
 - Tensorflow
 - Keras
 - Google Colab(If you don't have GPU then you can use Colab Notebook)

## Result/Performance

I got accuracy of ```96.91%``` on training data and ```96.65%``` on validation data after running model for 50 epochs.

![](images/malaria_model.PNG)
 
## Features

## Usage

## Credits
