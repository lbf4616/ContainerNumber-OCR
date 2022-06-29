# ContainerNumber-OCR
[![tf](https://img.shields.io/badge/TensorFlow-1.15.0-yellow?logo=tensorflow)](https://tensorflow.org/)  

Detection and Recognition Container number.  
 __Detection:__  Pixel_link + MobileNet_v2  
 __Recognition:__  LSTM  
 __Speed:__  RTX 2080Ti 200ms/image(1920*1080)  
 
![Sample](https://github.com/lbf4616/ContainerNumber-OCR/blob/master/Sample.png)
 
## Demo
1. Download pretrained model from [Google Drive](https://drive.google.com/open?id=18IGl5jOsUX4S6fKLHlw41JXEn4RRxIIF)  
2. Install requirements  
```
pip install -r requirements.txt
```
3. run
```
python containernumber_test_ckpt.py
```

## Dataset
[Google Drive](https://drive.google.com/drive/folders/13LpHEeFExmDJnw_U9peqLR-8uAAUMEzi?usp=sharing)

## Convert to Openvino Model
```
python tools/export.py \
  --checkpoint ckpt/recognition_v/model_all.ckpt-146000 \
  --output_dir recognition_v_ov
```
