import os
import cv2
import time
import tensorflow as tf
import numpy as np
from glob import glob
from detection_test_pb import detection
from recognition_test_pb import recognition
#from text_recognition.modelo import TextRecognition

from format_prech import revise
#from accuracy import acc

detection_model_path = "pb_models/detection.pb"
recognition_model_h_path = "pb_models/recognition_h.pb"
recognition_model_v_path = "pb_models/recognition_v.pb"
#acc('/home/blin/containernumber_result.txt')
# def get_all_layernames():
#     """get all layers name"""

#     pb_file_path = recognition_model_h_path

#     from tensorflow.python.platform import gfile

#     sess = tf.Session()
#     # with gfile.FastGFile(pb_file_path + 'model.pb', 'rb') as f:
#     with gfile.FastGFile(pb_file_path, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         sess.graph.as_default()
#         tf.import_graph_def(graph_def, name='')

#         tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
#         for tensor_name in tensor_name_list[:2]:
#             print(tensor_name, '\n')
# get_all_layernames()
#detection_model_path = "/home/blin/Downloads/text_detection/modelgpu2b8_new_f_0.0001/weights/model-24.save_weights"
#recognition_model_h_path = "/home/blin/Downloads/recognition_h/model_all.ckpt-8000"
#recognition_model_h_path = "/home/blin/Downloads/recognition_v/model_all.ckpt-90000"
#recognition_model_v_path = "/home/blin/Downloads/recognition_v/model_all.ckpt-90000"

with tf.Graph().as_default():
    detection_graph_def = tf.GraphDef()
    with open(detection_model_path, "rb") as f:
        detection_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(detection_graph_def, name="")

    sess_d = tf.Session()
    init = tf.global_variables_initializer()
    sess_d.run(init)
    input_x = sess_d.graph.get_tensor_by_name("Placeholder:0")
    segm_logits = sess_d.graph.get_tensor_by_name("model/segm_logits/add:0")
    link_logits = sess_d.graph.get_tensor_by_name("model/link_logits/Reshape:0")

with tf.Graph().as_default():
    recogniton_graph_def = tf.GraphDef()
    with open(recognition_model_h_path, "rb") as f:
        recogniton_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(recogniton_graph_def, name="")

    sess_r_h = tf.Session()
    init = tf.global_variables_initializer()
    sess_r_h.run(init)
    input_h = sess_r_h.graph.get_tensor_by_name("Placeholder:0")
    #input_h = tf.placeholder(tf.float32, [4, 32, 240, 1])
    model_out_h = sess_r_h.graph.get_tensor_by_name("shadow/LSTMLayers/transpose:0")
    decoded_h, _ = tf.nn.ctc_beam_search_decoder(model_out_h, 60 * np.ones(4), merge_repeated=False)

with tf.Graph().as_default():
    recogniton_graph_def = tf.GraphDef()
    with open(recognition_model_v_path, "rb") as f:
        recogniton_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(recogniton_graph_def, name="")

    sess_r_v = tf.Session()
    init = tf.global_variables_initializer()
    sess_r_v.run(init)
    input_v = sess_r_v.graph.get_tensor_by_name("Placeholder:0")
    model_out_v = sess_r_v.graph.get_tensor_by_name("shadow/LSTMLayers/transpose:0")
    decoded_v, _ = tf.nn.ctc_beam_search_decoder(model_out_v, 80 * np.ones(4), merge_repeated=False)


res_txt = open('containernumber_result.txt', 'w')

impaths = glob('/home/blin/Downloads/text_detection/test/*.jpg')
config = {}
config['segm_conf_thr'] = 0.8
config['link_conf_thr'] = 0.8
config['min_area'] = 300
config['min_height'] = 10
i = 0
total_time1 = time.time()
for impath in impaths:
    #impath = '/home/blin/Downloads/text_detection/test/1-144110001-OCR-LB-C02.jpg'
    #impath = '/home/blin/Downloads/text_detection/test/1-122728001-OCR-LB-C02.jpg'
    imname = os.path.basename(impath)
    im = cv2.imread(impath)
    #print(impath)
    t1 = time.time()
    bboxs = detection(im, sess_d, input_x, segm_logits, link_logits, config)
    t2 = time.time()
    #print('detection_time: ', (t2-t1),'result', bboxs)
    #bboxs = ['792, 364, 792, 298, 923, 298, 923, 364\n', '972, 375, 972, 303, 1271, 303, 1271, 375\n', '972, 455, 972, 389, 1109, 389, 1109, 455\n']
    predicted = recognition(im, sess_r_h, sess_r_v , bboxs, (240, 32), input_h, input_v, model_out_h, model_out_v, decoded_h, decoded_v)
    predicted_r = revise(predicted)
    t3 = time.time()
    #print('recognition_time: ', (t3-t2),'result', predicted)
    i+=1
    print(imname, i, predicted_r)
    line = imname + ' ' + predicted_r + '\n'
    res_txt.write(line)
res_txt.close()
total_time2 = time.time()

print('total_time: ', (total_time2 - total_time1))
