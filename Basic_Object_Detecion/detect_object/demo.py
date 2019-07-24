"""

@file  : demo.py

@author: xiaolu

@time  : 2019-07-24

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from datahelper import label_map_util
from datahelper import visualization_utils as vis_util

# 模型和标签的路径
PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
PATH_TO_LABELS = 'ssd_mobilenet_v1_coco_2017_11_17/mscoco_label_map.pbtxt'

NUM_CLASSES = 90  # 总共有90个类别

detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        od_graph_def.ParseFromString(fid.read())   # 直接把模型解析到我们当前这个图中
        tf.import_graph_def(od_graph_def, name='')

# 加载标签, 种类数
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)   # 进一步整理　{id1: name1, id2: name2....}


def load_image_into_numpy_array(image):
    '''
    将带检测的图片转为numpy数组
    :param image: 图片
    :return: 对应的numpy数组
    '''
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# 找两张测试图片
TEST_IMAGE_PATHS = ['test_images/image1.jpg', 'test_images/image2.jpg', 'test_images/street.jpg']

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # 1. 获取的输入口
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # 2. 最后预测的框
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # 3. 预测类别的得分
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # 4. 类别
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # 5. 当前图片中检测物体的个数
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )
            vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                               np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32),
                                                               np.squeeze(scores),
                                                               category_index,
                                                               use_normalized_coordinates=True,
                                                               line_thickness=8)

            plt.figure(figsize=(12, 8))
            plt.imshow(image_np)
            plt.show()
