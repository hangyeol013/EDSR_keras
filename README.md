# EDSR_keras



A implementation of the original paper ['Enhanced Deep Residual Networks for Single Image Super-Resolution'](https://arxiv.org/abs/1707.02921)



<center><img width = "400" src="https://user-images.githubusercontent.com/58276840/95935196-a628fd80-0e0d-11eb-9549-1bf61f1667a1.png"></center>



tensorflow-gpu 2.0.0, keras 2.3.1 based implementation on Python 3.6.9, using Jupyter Notebook.

(I converted the .ipynb file to .py file)




### Implementation
-------------------------------------------------------
My implementation may have some differences with the original paper:


#### Networks)

- Input image size: 3 (RGB)
- kernel size: 3x3
- Zero-padding for all layers
- EDSR_baseline) filters: 64, residual blocks: 16
- EDSR) filters: 256, residual blocks: 32
- Upsampling: Shuffle (tensorflow depth_to_space)


#### Training)

- Loss Function: MAE (Mean Absolute Error, L1)
- Optimizer: Adam
- Learning rate: 10e-4 (Decreased by half at every 100 epochs) 
- Batch size: 16


#### Dataset)

##### Training)
- DIV2K (800 images) (Training set)
- Another 20 images from the validation set of the BSD500 dataset (Validation set)
- Data augmentation (flip: vertical, horizontal, vertical/horizontal, transpose): 7 times more images for training
- Patch size: 48 (Input), 48 x scale (Label)


##### Test)
- Set5, Set14, BSD200, Urban100



### Use
-------------------------------------------------------

You can generate dataset (training sample, test sample) through matlab files in Dataset_Generating directory
- Excute for data augmentation: `EDSR_data_aug.m`
- Excute for making LR images: `Generate_LR.m`
- Excute for training sample: `EDSR_generate_train.m`
- Excute for test sample: `EDSR_generate_test.m`


I also uploaded the trained weight files.

With EDSR.ipynb file and weight files in 'weight', you can test the network.
I uploaded EDSR model as well but I couldn't upload the weight files for them because of memory issue.
Here you can only test the baseline version of EDSR
(or you can train the EDSR models by yourself and test them)

- Execute for x2 EDSR_baseline: `EDSR_baseline_scale2`, `/weights/EDSR_checkpoint_baseline_scale2.h5`
- Execute for x3 EDSR_baseline: `EDSR_baseline_scale3`, `/weights/EDSR_checkpoint_baseline_scale3.h5`
- Execute for x4 EDSR_baseline: `EDSR_baseline_scale4`, `/weights/EDSR_checkpoint_baseline_scale4.h5`
