import tensorflow as tf
import numpy as np
import cv2
import os
import time
def get_neighbours(x_coord, y_coord):
    """ Returns 8-point neighbourhood of given point. """

    return [(x_coord - 1, y_coord - 1), (x_coord, y_coord - 1), (x_coord + 1, y_coord - 1), \
            (x_coord - 1, y_coord), (x_coord + 1, y_coord), \
            (x_coord - 1, y_coord + 1), (x_coord, y_coord + 1), (x_coord + 1, y_coord + 1)]

def is_valid_coord(x_coord, y_coord, width, height):
    """ Returns true if given point inside image frame. """

    return 0 <= x_coord < width and 0 <= y_coord < height
           
def decode_image(segm_scores, link_scores, segm_conf_threshold, link_conf_threshold):
    """ Convert softmax scores to mask. """

    segm_mask = segm_scores >= segm_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    points = list(zip(*np.where(segm_mask)))
    height, width = np.shape(segm_mask)
    group_mask = dict.fromkeys(points, -1)

    def find_parent(point):
        return group_mask[point]

    def set_parent(point, parent):
        group_mask[point] = parent

    def is_root(point):
        return find_parent(point) == -1

    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True

        if update_parent:
            set_parent(point, root)

        return root

    def join(point1, point2):
        root1 = find_root(point1)
        root2 = find_root(point2)

        if root1 != root2:
            set_parent(root1, root2)

    def get_all():
        root_map = {}

        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]

        mask = np.zeros_like(segm_mask, dtype=np.int32)
        for point in points:
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask

    for point in points:
        y_coord, x_coord = point
        neighbours = get_neighbours(x_coord, y_coord)
        for n_idx, (neighbour_x, neighbour_y) in enumerate(neighbours):
            if is_valid_coord(neighbour_x, neighbour_y, width, height):
                link_value = link_mask[y_coord, x_coord, n_idx]
                segm_value = segm_mask[neighbour_y, neighbour_x]
                if link_value and segm_value:
                    join(point, (neighbour_y, neighbour_x))

    mask = get_all()

    return mask


def rect_to_xys(rect, image_shape):
    """ Converts rotated rectangle to points. """

    height, width = image_shape[0:2]

    def get_valid_x(x_coord):
        return np.clip(x_coord, 0, width - 1)

    def get_valid_y(y_coord):
        return np.clip(y_coord, 0, height - 1)

    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x_coord, y_coord) in enumerate(points):
        x_coord = get_valid_x(x_coord)
        y_coord = get_valid_y(y_coord)
        points[i_xy, :] = [x_coord, y_coord]
    points = np.reshape(points, -1)
    return points


def min_area_rect(contour):
    """ Returns minimum area rectangle. """

    (center_x, cencter_y), (width, height), theta = cv2.minAreaRect(contour)
    return [center_x, cencter_y, width, height, theta], width * height

def softmax(logits):
    """ Returns softmax given logits. """

    max_logits = np.max(logits, axis=-1, keepdims=True)
    numerator = np.exp(logits - max_logits)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator

def mask_to_bboxes(mask, config, image_shape):
    """ Converts mask to bounding boxes. """

    image_h, image_w = image_shape[0:2]

    min_area = config['min_area']
    min_height = config['min_height']

    bboxes = []
    max_bbox_idx = mask.max()
    mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

    for bbox_idx in range(1, max_bbox_idx + 1):
        bbox_mask = (mask == bbox_idx).astype(np.uint8)
        cnts = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)

        box_width, box_height = rect[2:-1]
        if min(box_width, box_height) < min_height:
            continue

        if rect_area < min_area:
            continue

        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)

    return bboxes

def decode_batch(segm_scores, link_scores, config):
    """ Returns boxes mask for each input image in batch."""

    batch_size = segm_scores.shape[0]
    batch_mask = []

    for image_idx in range(batch_size):
        image_pos_pixel_scores = segm_scores[image_idx, :, :]
        image_pos_link_scores = link_scores[image_idx, :, :, :]
        mask = decode_image(image_pos_pixel_scores, image_pos_link_scores,
                            config['segm_conf_thr'], config['link_conf_thr'])
        batch_mask.append(mask)
    
    return np.asarray(batch_mask, np.int32)

def to_boxes_any(image_data, segm_pos_scores, link_pos_scores, conf):
    """ Returns boxes for each image in batch. """
    def write_result_as_txt(bboxes):
        lines = []
        for bbox in enumerate(bboxes):
            values = [int(v) for v in bbox[1]]
            line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
            lines.append(line)
        return lines
    
    mask = decode_batch(segm_pos_scores, link_pos_scores, conf)[0, ...]
    
    bboxes = mask_to_bboxes(mask, conf, image_data.shape)
    #txt_path = os.path.join(dataset,'txt')
    
    lines = write_result_as_txt(bboxes)
    return lines

def detection(img, session_d, input_x, segm_logits, link_logits, config):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img.astype(np.float32)
    image = cv2.resize(image, (1280,768))

    segm_logits,link_logits = session_d.run([segm_logits,link_logits], feed_dict={input_x:np.reshape(image, [1, 768, 1280, 3])})
    segm_scores = softmax(segm_logits)
    link_scores = softmax(link_logits)


    bboxs = to_boxes_any(img, segm_scores[:, :, :, 1], link_scores[:, :, :, :, 1], config)

    return bboxs

#[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
# result = summarize_graph("/home/blin/Downloads/text_detection/tools/detection.pb")
# print(result)
#detection("/home/blin/Downloads/text_detection/test/1-122700001-OCR-RF-D01.jpg", "/home/blin/Downloads/text_detection/tools/detection.pb", "/home/blin/Downloads/text_detection/1-122700001-OCR-RF-D01.txt")

# from tensorflow.python.framework import tensor_util
# from google.protobuf import text_format
# import tensorflow as tf
# from tensorflow.python.platform import gfile
# from tensorflow.python.framework import tensor_util


# GRAPH_PB_PATH = '/home/blin/Downloads/text_detection/tools/detection.pb' #path to your .pb file
# with tf.Session() as sess:
# 	print("load graph")
# 	with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
# 		graph_def = tf.GraphDef()
#     # Note: one of the following two lines work if required libraries are available
# 		#text_format.Merge(f.read(), graph_def)
# 		graph_def.ParseFromString(f.read())
# 		tf.import_graph_def(graph_def, name='')
# 		for i,n in enumerate(graph_def.node):
# 			print("Name of the node - %s" % n.name)
