# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class Model:
    def __init__(self,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 32 * 32 * 3])
        self.y_ = tf.placeholder(tf.int32, [None])

        # TODO:  fill the blank of the arguments
        self.loss, self.pred, self.acc = self.forward(True, reuse=tf.AUTO_REUSE)
        self.loss_val, self.pred_val, self.acc_val = self.forward(False, reuse=tf.AUTO_REUSE)
        
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()


        # TODO:  maybe you need to update the parameter of batch_normalization?
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([self.train_op, self.update_ops])

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse):
    
        with tf.variable_scope("model", reuse=reuse):
            # TODO:  implement input -- Linear -- BN -- ReLU -- Dropout -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # print(self.x_)
            tensor = self.x_
            # Your Linear Layer
            tensor = tf.layers.dense(tensor, units=1000, name='dense1', reuse=reuse)
            # Your BN Layer: use batch_normalization_layer function
            # tensor = batch_normalization_layer(tensor, "bn1", reuse=reuse, is_train=is_train)
            # Your Relu Layer
            tensor = tf.nn.relu(tensor)
            # Your Dropout Layer: use dropout_layer function
            tensor = dropout_layer(tensor, FLAGS.drop_rate, "dropout1", is_train=is_train)
            # Your Linear Layer
            tensor = tf.layers.dense(tensor, units=10, name='dense2', reuse=reuse)
            logits = tensor
            # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch
        
        return loss, pred, acc

def batch_normalization_layer(incoming, name, reuse, is_train):
    # TODO: implement the batch normalization function and applied it on fully-connected layers
    # NOTE:  If isTrain is True, you should use mu and sigma calculated based on mini-batch
    #       If isTrain is False, you must use mu and sigma estimated from training data
    return tf.layers.batch_normalization(incoming, name=name, reuse=reuse, training=is_train)
    
def dropout_layer(incoming, drop_rate, name, is_train):
    # TODO: implement the dropout function and applied it on fully-connected layers
    # Note: When drop_rate=0, it means drop no values
    #       If isTrain is True, you should randomly drop some values, and scale the others by 1 / (1 - drop_rate)
    #       If isTrain is False, remain all values not changed
    return tf.layers.dropout(incoming, rate=drop_rate, name=name, training=is_train)