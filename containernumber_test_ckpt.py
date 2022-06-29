import cv2
import time
import tensorflow as tf
from glob import glob
from detection_test_pb import detection
from recognition_test_pb import recognition
from text_recognition.model import TextRecognition
import os
import numpy as np
from tqdm import tqdm

detection_model_path = "pb_models/detection.pb"

recognition_model_h_path = "ckpt/recognition_h/model_all.ckpt-8000"
recognition_model_v_path = "ckpt/recognition_v/model_all.ckpt-146000"

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

bs = 4
model = TextRecognition(is_training=False, num_classes=37)

images_ph_h = tf.placeholder(tf.float32, [bs, 32, 240, 1])
model_out_h = model(inputdata=images_ph_h)
saver_h = tf.train.Saver()
sess_r_h=tf.Session()
saver_h.restore(sess=sess_r_h, save_path=recognition_model_h_path)
decoded_h, _ = tf.nn.ctc_beam_search_decoder(model_out_h, 60 * np.ones(bs), merge_repeated=False)

with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    images_ph_v = tf.placeholder(tf.float32, [bs, 32, 320, 1])
    model_out_v = model(inputdata=images_ph_v)
    saver_v = tf.train.Saver()
    sess_r_v=tf.Session()
    saver_v.restore(sess=sess_r_v, save_path=recognition_model_v_path)
    decoded_v, _ = tf.nn.ctc_beam_search_decoder(model_out_v, 80 * np.ones(bs), merge_repeated=False)

res_txt = open('containernumber_result.txt', 'w')
impaths = glob('samples/*.jpg')
res_dir = "output"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

config = {}
config['segm_conf_thr'] = 0.8
config['link_conf_thr'] = 0.8
config['min_area'] = 300
config['min_height'] = 10

total_time1 = time.time()  
for impath in tqdm(impaths):

    imname = os.path.basename(impath)
    im = cv2.imread(impath)
    t1 = time.time()
    bboxs = detection(im, sess_d, input_x, segm_logits, link_logits, config)
    for bbox in bboxs:
        pts = [int(p) for p in bbox.split(",")]
        cv2.rectangle(im, (pts[0], pts[1]), (pts[4], pts[5]), (0, 255, 0), 2)
    t2 = time.time()
    print('detection_time: ', (t2-t1),'result', bboxs)
    predicted = recognition(im, sess_r_h, sess_r_v , bboxs, (240, 32), images_ph_h, images_ph_v, model_out_h, model_out_v, decoded_h, decoded_v)
    cv2.putText(im, predicted, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    t3 = time.time()
    print('recognition_time: ', (t3-t2),'result', predicted)
    cv2.imwrite(os.path.join("output", imname), im)
    line = imname + ' ' + predicted + '\n'
    res_txt.write(line)
res_txt.close()
total_time2 = time.time()
print('total_time: ', (total_time2 - total_time1))

# from accuracy import acc
# acc('containernumber_result.txt')
