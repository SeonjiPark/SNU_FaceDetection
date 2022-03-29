# SNU_RetinaFace

# reference paper


# Environments
Pytorch 1.7.0
CUDA 11.2 & cuDNN 7.6.5
Python 3.8.8

You can type the following command to easily build the environment. 
Download 'retinaface.yml' and type the following command.

conda env create -f retinaface.yml


# Dataset 다운 주소
train/val/test dataset - widerface

http://shuoyang1213.me/WIDERFACE/


# Directory
|── experiments
    ├──> experiment_name 
         ├──> ckpt : trained models will be saved here
         └──> log  : log will be saved here
|── dataset
    ├──> dataset_name1 
         ├──> train : training images of dataset_name1 should be saved here
         └──> test  : test images of dataset_name1 should be saved here
    ├──> dataset_name2
         ├──> train : training images of dataset_name2 should be saved here
         └──> test  : test images of dataset_name2 should be saved here         
|── utils : files for utility functions
|── config.py : configuration should be controlled only here 
|── decode.py : decode compressed files to images
|── encode.py : encode images to compressed format
|── fdnet_env.yml : virtual enviornment specification
|── model.py : architecture of FDNet
|── test.py : test the model. performance is estimated (not actual compression)
|── train.py : train the model
└── jpegxl : folder for jpegxl library. explained below.



# Guidelines for implementation

=== Our Code ===
학습용 코드 - train.py
테스트용 코드 - test.py (GT 존재해서 AP 측정 가능할 때)
테스트용 코드2 - inference.py (GT 존재하지 않아서 AP 측정 불가능)


=== 학습된 ckpt ===

구글 드라이브 주소 : https://drive.google.com/drive/folders/1bbxIfmmlhs33uBkTasL6ksnPfabFFpNI?usp=sharing


=== Official Code Evaluation ===

Official Code 사이트 : https://github.com/biubug6/Pytorch_Retinaface

official_evaluate/test_official.py 실행
