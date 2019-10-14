# import the necessary packages
import argparse
import cv2
def click_and_crop(event, x, y, flags, param):
 # grab references to the global variables
    global refPt, cropping
    
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    
    if event == cv2.EVENT_LBUTTONDOWN :
        print(cods)
        refPt = [x, y]
        cods.append(refPt)
        print(cods[0][1],y)
        print(len(cods))
        if len(cods) == 4:
            line = str(cods[0][0]) + ',' + str(cods[0][1]) + ',' + str(cods[1][0]) + ',' + str(cods[1][1]) + ',' + str(cods[2][0]) + ',' + str(cods[2][1]) + ',' + str(cods[3][0]) + ',' + str(cods[3][1]) + ','
            label = input("请输入label：")
            line = line + label + '\n'
            f.write(line)
            cods.clear()           
            print(label)

            
from glob import glob
impaths = glob('/home/blin/Downloads/newim_part2/*.jpg')
for impath in impaths[:1]:
    cods = []
    impath = '/home/blin/Videos/car-train/images/1-124126001-OCR-LB-C02.jpg'
    image = cv2.imread(impath)
    f = open(impath.replace('jpg','txt'),'w')
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    while(1):
        cv2.imshow('image',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    

    f.close()
