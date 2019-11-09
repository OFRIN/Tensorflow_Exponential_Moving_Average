# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import time
import numpy as np
import tensorflow as tf

from Define import *
from WideResNet import *
from Utils import *
from DataAugmentation import *

# 1. dataset
test_data_list = np.load('./dataset/test.npy', allow_pickle = True)
test_iteration = len(test_data_list) // BATCH_SIZE

# 2. model
input_var = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
label_var = tf.placeholder(tf.float32, [None, CLASSES])
is_training = tf.placeholder(tf.bool)

logits_op, predictions_op = WideResNet(input_var, is_training, reuse = False)

correct_op = tf.equal(tf.argmax(predictions_op, axis = -1), tf.argmax(label_var, axis = -1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

# 3. test
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/EMA_100%.ckpt')

test_time = time.time()
test_accuracy_list = []

for i in range(test_iteration):
    batch_data_list = test_data_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

    batch_image_data = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype = np.float32)
    batch_label_data = np.zeros((BATCH_SIZE, CLASSES), dtype = np.float32)
    
    for i, (image, label) in enumerate(batch_data_list):
        batch_image_data[i] = image.astype(np.float32)
        batch_label_data[i] = label.astype(np.float32)
    
    _feed_dict = {
        input_var : batch_image_data,
        label_var : batch_label_data,
        is_training : False
    }
    accuracy = sess.run(accuracy_op, feed_dict = _feed_dict)
    test_accuracy_list.append(accuracy)

test_time = int(time.time() - test_time)
test_accuracy = np.mean(test_accuracy_list)

# test_accuracy = 92.74, test_time = 8sec
print('test_accuracy = {:.2f}, test_time = {}sec'.format(test_accuracy, test_time))
