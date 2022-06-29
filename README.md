# ContainerNumber-OCR
Detection and Recognition Container number.  
 __Detection:__  Pixel_link + MobileNet_v2  
 __Recognition:__  LSTM  
 __Speed:__  RTX 2080Ti 200ms/image(1920*1080)  
 
![Sample](https://github.com/lbf4616/ContainerNumber-OCR/blob/master/Sample.png)
 
## Pretrained model
Including Pb models and ckpt  
[Google Drive](https://drive.google.com/open?id=18IGl5jOsUX4S6fKLHlw41JXEn4RRxIIF)  

## Dataset
[Google Drive](https://drive.google.com/drive/folders/13LpHEeFExmDJnw_U9peqLR-8uAAUMEzi?usp=sharing)

## Convert to Openvino Model
```
python tools/export.py \
  --checkpoint ckpt/recognition_v/model_all.ckpt-146000 \
  --output_dir recognition_v_ov
```
