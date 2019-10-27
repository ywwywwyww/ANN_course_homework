"""
CNN model(s)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

FLAGS = tf.app.flags.FLAGS

def CNN_forward(input, length, embed_units, filter_sizes, num_filters, is_train=None, reuse=None):
    pooled = []
    for filter_size in filter_sizes:
        with tf.name_scope("CNN-%d" % filter_size):
            conv = tf.layers.conv2d(input, num_filters, (filter_size, embed_units), padding='valid', name="conv2d_1_%d" % filter_size, reuse=reuse)
                # shape: [batch, length-filter_size+1, 1, num_filters]
            # print(conv.shape)
            pool = tf.reduce_max(conv, 1)
            # print(pool.shape)
            # pool = tf.layers.max_pooling2d(conv, (length - filter_size + 1, 1), (1, 1), name="maxpooling2d_1")
            reshaped = tf.reshape(pool, shape=[-1, num_filters])
            pooled.append(reshaped)

    with tf.name_scope("CNN"):
        tensor = tf.concat(pooled, 1)
        dropout = tf.layers.dropout(tensor, FLAGS.drop_rate, name="dropout_1", training=is_train)
        logits = tf.layers.dense(dropout, units=FLAGS.labels, name="dense_1", reuse=reuse)

    return logits
