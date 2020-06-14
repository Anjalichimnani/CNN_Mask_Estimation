# Mask Estimation using CNN

The Office Dataset created in [Git Hub](https://github.com/Anjalichimnani/EVA4_Custom_Data) is used for Image Segmentation. The images are [Background Images](https://github.com/Anjalichimnani/EVA4_Custom_Data/blob/master/reference_images/bg_images.png), [Foreground Images](https://github.com/Anjalichimnani/EVA4_Custom_Data/blob/master/reference_images/fg_images.png) and [Masks](https://github.com/Anjalichimnani/EVA4_Custom_Data/blob/master/reference_images/mask_images.png)

The Image segmentation is performed using three different architectures: 
* Basic Architecture
    5 Convolution Layers and 1 Max Pool which takes the concatenation on FG_BG_Images and BG Images. 

* Customized Net 
    6 Convolution Layers which takes 2 inputs of FG_BG_Images and BG_Images and performs concatenation after 2 convolutions
    
* UNet
    Advanced architecture performing an Encoder/Decoder operation on the image. The convolution is performed and to obtain the output, upsampling combining the features at same levels is performed and consequently, convoluted to desired channels
    
    ![UNet](https://github.com/Anjalichimnani/EVA4_Custom_Data/blob/master/reference_images/mask_images.png)
    
The Masks generted based on the different architectures are as below: 

## Basic Architecture
Predicted

Actual

## Customized Architecture
Predicted

Actual

## UNet Architecture
Predicted

Actual

## References
[UNet](https://towardsdatascience.com/u-net-b229b32b4a71)
[Custom Data](https://github.com/Anjalichimnani/EVA4_Custom_Data)