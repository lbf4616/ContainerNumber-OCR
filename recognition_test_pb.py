import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
from text_recognition.dataset import Dataset
from format_prech import revise
import time
def order_points_old(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect

def affine(im, pts, r_size, direction):
  
    bbox = [[pts[0],pts[1]],[pts[2],pts[3]],[pts[4],pts[5]],[pts[6],pts[7]]]
    rect = order_points_old(np.array(bbox))

    if rect[1][1]  < rect[0][1] and abs(rect[1][1] - rect[0][1])> 100:
        
        rect[[0,1,2,3],:] = rect[[1,2,3,0],:]
    pts1 = np.float32([rect[0], rect[1], rect[2]])
    if direction == 0:
        w, h = r_size
        w = 240
    else:
        h, w = r_size
        h = 320

    pts2 = np.float32([[0,0], [w-1,0], [w-1, h-1]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(im, M, (w, h))
    

    return dst

def sort_box(box_file):

    res = []
    direction = -1
    count_v = 0
    count_h = 0

    listx = []
    listy = []
    listx2 = []
    listy2 = []
    boxlist = []

    for box in box_file:
        box = [int(i) for i in box.rstrip().split(", ")]
        xmin = np.min(np.array(box[::2]))
        xmax = np.max(np.array(box[::2]))
        ymin = np.min(np.array(box[1::2]))
        ymax = np.max(np.array(box[1::2]))
        
        listx.append(int(xmin))
        listy.append(int(ymin))

        listx2.append(int(xmax))
        listy2.append(int(ymax))

        boxlist.append(box)

        w = xmax - xmin
        h = ymax - ymin
        if w > h:
            count_h += 1
        else:
            count_v +=1
    
    if count_h > count_v:
        direction = 0
        df = pd.DataFrame({'x':listx,'y':listy,'x2':listx2,'y2':listy2,"bboxes":boxlist})
    else:
        direction = 1
        df = pd.DataFrame({'y':listx,'x':listy,'y2':listx2,'x2':listy2,"bboxes":boxlist})
        
    #check wheher in same line
    df = df.sort_values(['y2','x'])
    temp1 =0 
    dfidx = 0
    listline = []
    line_threshold = 25
    for index, row in df.iterrows():
        y1 = row["y"]
        
        if dfidx != 0 and abs(temp1 - y1) > line_threshold:
             newlistline = sorted(listline, key = lambda k:k['x'])
             res.extend(newlistline)
             listline = []
        temp1 = y1
        dfidx+=1
        dictbox = {"x":row["x"],"bboxes":row["bboxes"]}
        listline.append(dictbox)
    newlistline = sorted(listline, key = lambda k:k['x'])
    res.extend(newlistline)
    return res, direction

def recognition(img, session_r_h, session_r_v, bboxs, r_size, images_ph_h, images_ph_v, model_out_h, model_out_v, decoded_h, decoded_v):
    int_to_char=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    boxlist, direction = sort_box(bboxs)

    result = ''
    num = len(boxlist)
    idx = 0
    for idx, item in enumerate(boxlist):
        ori_box = item['bboxes']
        img_affined = affine(img, ori_box, r_size, direction)

        img_affined = cv2.cvtColor(img_affined, cv2.COLOR_BGR2GRAY)
        #img_affined = cv2.resize(img_affined,(w, h))
        if direction == 1:
            img_affined = img_affined.transpose((1,0))
            img_affined = img_affined[::-1]
            #sq = 80
            # cv2.imshow('2',img_affined)
            # cv2.waitKey()

        img_affined = np.array([img_affined])
        img_affined = np.expand_dims(img_affined, axis = 3)
        if idx == 0 :
            img_affined_final = img_affined
        if idx != 0 and idx < 4:
            img_affined_final = np.concatenate((img_affined_final, img_affined), 0)

        if idx == 4 :
            img_affined_final2 = img_affined
        if idx > 4:
            img_affined_final2 = np.concatenate((img_affined_final2, img_affined), 0)
        # tt1 = time.time()
        # decoded, _ = tf.nn.ctc_beam_search_decoder(model_out, sq * np.ones(1), merge_repeated=False)
        # tt2 = time.time()
    if idx < 3:
        for j in range(3-idx):
            img_affined_final = np.concatenate((img_affined_final, img_affined), 0)

    if idx > 3 and idx <7:
         for j in range(4, 7-idx):
            img_affined_final2 = np.concatenate((img_affined_final2, img_affined), 0)
    if idx < 4:
        if direction == 0:
            preds, _ = session_r_h.run([decoded_h, model_out_h], feed_dict={images_ph_h: img_affined_final})
        else:
            preds, _ = session_r_v.run([decoded_v, model_out_v], feed_dict={images_ph_v: img_affined_final})

        for i in range (0,num):
            try:
                predicted = Dataset.sparse_tensor_to_str(preds[0], int_to_char)[i]
                result += predicted
            except:
                predicted = ''
                result += predicted

    else:
        if direction == 0:
            preds1, _ = session_r_h.run([decoded_h, model_out_h], feed_dict={images_ph_h: img_affined_final})
            preds2, _ = session_r_h.run([decoded_h, model_out_h], feed_dict={images_ph_h: img_affined_final2})
        else:
            preds1, _ = session_r_v.run([decoded_v, model_out_v], feed_dict={images_ph_v: img_affined_final})
            preds2, _ = session_r_v.run([decoded_v, model_out_v], feed_dict={images_ph_v: img_affined_final2})

        for i in range (0,3):
            try:
                predicted = Dataset.sparse_tensor_to_str(preds1[0], int_to_char)[i]
                result += predicted
            except:
                predicted = ''
                result += predicted

        for i in range (4,num):

            try:
                predicted = Dataset.sparse_tensor_to_str(preds2[0], int_to_char)[i]
                result += predicted
            except:
                predicted = ''
                result += predicted
    
        #result += predicted

    return revise(result)
            

#[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
# result = summarize_graph("/home/blin/Downloads/text_detection/tools/detection.pb")
# print(result)

# predicted = recognition("/home/blin/Downloads/text_recognition/chim/1-122728001-OCR-LB-C02.jpg_001.jpg", '/home/blin/Downloads/text_recognition/tools/recognition_h.pb')
# print(predicted)