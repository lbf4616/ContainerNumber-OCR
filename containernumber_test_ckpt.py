import cv2
import time
import tensorflow as tf
from glob import glob
from detection_test_pb import detection
from recognition_test_pb_copy import recognition
#from recognition_test_pb_b1 import recognition
from text_recognition.model import TextRecognition
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from accuracy import acc
detection_model_path = "/home/blin/Downloads/text_detection/pb_models/detection.pb"
#recognition_model_h_path = "/home/blin/Downloads/text_detection/pb_models/recognition_h.pb"
#recognition_model_v_path = "/home/blin/Downloads/text_detection/pb_models/recognition_v.pb"


#detection_model_path = "/home/blin/Downloads/text_detection/modelgpu2b8_new_f_0.0001/weights/model-24.save_weights"
recognition_model_h_path = "/home/blin/Downloads/recognition_h/model_all.ckpt-8000"
#recognition_model_h_path = "/home/blin/Downloads/recognition_v/model_all.ckpt-90000"
recognition_model_v_path = "/home/blin/Downloads/recognition_v/model_all.ckpt-90000"

#recognition_model_h_path = "/home/blin/Downloads/train/text_recognition/model/2019-10-13-09-47-07/model_all.ckpt-430000"
#recognition_model_h_path = "/home/blin/Downloads/recognition_v/model_all.ckpt-90000"
recognition_model_v_path = "/home/blin/Downloads/train/text_recognition/model/2019-10-13-09-57-12/model_all.ckpt-146000"
with tf.Graph().as_default():
    detection_graph_def = tf.GraphDef()
    with open(detection_model_path, "rb") as f:
        detection_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(detection_graph_def, name="")

    sess_d=tf.Session()
    init = tf.global_variables_initializer()
    sess_d.run(init)
    input_x = sess_d.graph.get_tensor_by_name("Placeholder:0")
    segm_logits = sess_d.graph.get_tensor_by_name("model/segm_logits/add:0")
    link_logits = sess_d.graph.get_tensor_by_name("model/link_logits/Reshape:0")
'''
with tf.Graph().as_default():
    recogniton_graph_def = tf.GraphDef()
    with open(recognition_model_h_path, "rb") as f:
        recogniton_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(recogniton_graph_def, name="")

    sess_r_h=tf.Session()
    #init = tf.global_variables_initializer()
    sess_r_h.run(init)

with tf.Graph().as_default():
    recogniton_graph_def = tf.GraphDef()
    with open(recognition_model_v_path, "rb") as f:
        recogniton_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(recogniton_graph_def, name="")

    sess_r_v=tf.Session()
    #init = tf.global_variables_initializer()
    sess_r_v.run(init)
'''
bs = 4
model = TextRecognition(is_training=False, num_classes=37)

images_ph_h = tf.placeholder(tf.float32, [bs, 32, 240, 1])
model_out_h = model(inputdata=images_ph_h)
saver_h = tf.train.Saver()
sess_r_h=tf.Session()
saver_h.restore(sess=sess_r_h, save_path=recognition_model_h_path)
decoded_h, _ = tf.nn.ctc_beam_search_decoder(model_out_h, 60 * np.ones(bs), merge_repeated=False)

images_ph_v = tf.placeholder(tf.float32, [bs, 32, 320, 1])
model_out_v = model(inputdata=images_ph_v)
saver_v = tf.train.Saver()
sess_r_v=tf.Session()
saver_v.restore(sess=sess_r_v, save_path=recognition_model_v_path)
decoded_v, _ = tf.nn.ctc_beam_search_decoder(model_out_v, 80 * np.ones(bs), merge_repeated=False)

res_txt = open('/home/blin/containernumber_result.txt', 'w')
impath = "/home/blin/Downloads/text_detection/test/1-122738001-OCR-RF-D01.jpg"
impaths = glob('/home/blin/Downloads/text_detection/test/*.jpg')
config = {}
config['segm_conf_thr'] = 0.8
config['link_conf_thr'] = 0.8
config['min_area'] = 300
config['min_height'] = 10
i = 0
total_time1 = time.time()  
for impath in impaths:
    #impath = '/home/blin/Downloads/text_detection/test/1-123152001-OCR-LF-C01.jpg'
    #impath = '/home/blin/Downloads/text_detection/test/1-142434001-OCR-AH-A01.jpg'
    imname = os.path.basename(impath)
    im = cv2.imread(impath)
    print(impath)
    t1 = time.time()
    bboxs = detection(im, sess_d, input_x, segm_logits, link_logits, config)
    t2 = time.time()
    print('detection_time: ', (t2-t1),'result', bboxs)
    #bboxs = ['792, 364, 792, 298, 923, 298, 923, 364\n', '972, 375, 972, 303, 1271, 303, 1271, 375\n', '972, 455, 972, 389, 1109, 389, 1109, 455\n']
    predicted = recognition(im, sess_r_h, sess_r_v , bboxs, (240, 32), images_ph_h, images_ph_v, model_out_h, model_out_v, decoded_h, decoded_v)
    #predicted = recognition(im, sess_r_h , bboxs, (240, 32), images_ph_h, model_out_h, decoded_h)
    t3 = time.time()
    print('recognition_time: ', (t3-t2),'result', predicted)
    i+=1
    print(i)
    line = imname + ' ' + predicted + '\n'
    res_txt.write(line)
res_txt.close()
total_time2 = time.time()
print('total_time: ', (total_time2 - total_time1))
acc('/home/blin/containernumber_result.txt')