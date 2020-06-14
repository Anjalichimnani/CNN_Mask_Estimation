# Mask Estimation using CNN

The Office Dataset created in [Git Hub](https://github.com/Anjalichimnani/EVA4_Custom_Data) is used for Image Segmentation. The images are [Background Images](https://github.com/Anjalichimnani/EVA4_Custom_Data/blob/master/reference_images/bg_images.png), [Foreground Images](https://github.com/Anjalichimnani/EVA4_Custom_Data/blob/master/reference_images/fg_images.png) and [Masks](https://github.com/Anjalichimnani/EVA4_Custom_Data/blob/master/reference_images/mask_images.png)

The Image segmentation is performed using three different architectures: 
* Basic Architecture
    5 Convolution Layers and 1 Max Pool which takes the concatenation on FG_BG_Images and BG Images. Input: (6, 64, 64)
    Code can be accessed at [Basic Net](https://github.com/Anjalichimnani/CNN_Mask_Estimation/blob/master/library/models/CustomNet.py)
    
    
    Total params: 523,521
    Trainable params: 523,521
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.09
    Forward/backward pass size (MB): 23.32
    Params size (MB): 2.00
    Estimated Total Size (MB): 25.41
    
    ![Customized Net](https://github.com/Anjalichimnani/CNN_Mask_Estimation/blob/master/reference/Basic_Net_Parameters.PNG)

* Customized Net 
    6 Convolution Layers which takes 2 inputs of FG_BG_Images and BG_Images and performs concatenation after 2 convolutions. 
    Code can be accessed at [Customized Net Net](https://github.com/Anjalichimnani/CNN_Mask_Estimation/blob/master/library/models/Net.py)
    
    ![Customized Net](https://github.com/Anjalichimnani/CNN_Mask_Estimation/blob/master/reference/Customized_Net_Parameters.PNG)
    
* UNet
    Advanced architecture performing an Encoder/Decoder operation on the image. The convolution is performed and to obtain the output, upsampling combining the features at same levels is performed and consequently, convoluted to desired channels. 
    Code can be accessed at [UNet](https://github.com/Anjalichimnani/CNN_Mask_Estimation/blob/master/library/models/UNet.py)
    
    ![UNet](https://github.com/Anjalichimnani/CNN_Mask_Estimation/blob/master/reference/u-net-architecture.png)
    
    
    Total params: 34,520,514
    Trainable params: 34,520,514
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 1.15
    Forward/backward pass size (MB): 367.97
    Params size (MB): 131.69
    Estimated Total Size (MB): 500.81
    
    ![UNet Parameters](https://github.com/Anjalichimnani/CNN_Mask_Estimation/blob/master/reference/UNet_Parameters.PNG)
    
    
* The Images taken for segmentation are placed at [Link](https://github.com/Anjalichimnani/CNN_Mask_Estimation/tree/master/reference) for reference. 
* The complete Code can be found at [Link](https://github.com/Anjalichimnani/CNN_Mask_Estimation/blob/master/CNN_Image_Segmentation.ipynb)
* The Custom OfficeDataSet can be found at [Loader](https://github.com/Anjalichimnani/CNN_Mask_Estimation/blob/master/library/dataloaders/custom_data_loader.py)
 
## Image Segmentation Results  
The Masks generted based on the different architectures are as below: 

### Basic Architecture
* Predicted

![Predicted Mask](https://github.com/Anjalichimnani/CNN_Mask_Estimation/blob/master/reference/Basic_Architecture_Predicted_Mask.PNG)


### UNet Architecture
* Predicted

![Predicted Mask](https://github.com/Anjalichimnani/CNN_Mask_Estimation/blob/master/reference/UNet_Predicted_Masks.PNG)


## References
[UNet](https://towardsdatascience.com/u-net-b229b32b4a71)

[Custom Data](https://github.com/Anjalichimnani/EVA4_Custom_Data)