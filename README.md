# ContainerNumber-OCR
Detection and Recognition Container number.  
 __Detection:__  Pixel_link + MobileNet_v2  
 __Recognition:__  LSTM  
 __Speed:__  RTX 2080Ti 200ms/image(1920*1080)  
 
## Pretrained model
Including Pb models and ckpt  
[Google Drive](https://drive.google.com/open?id=18IGl5jOsUX4S6fKLHlw41JXEn4RRxIIF)  

## Sample
![Sample](https://github.com/lbf4616/ContainerNumber-OCR/blob/master/Sample.png)

## Convert to Openvino Model
```
python tools/export.py \
  --checkpoint ckpt/recognition_v/model_all.ckpt-146000 \
  --output_dir recognition_v_ov
```
