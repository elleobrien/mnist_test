import os
import json
import time
import tensorflow.compat.v1 as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.disable_v2_behavior()

import mnist

dirname = os.path.dirname(__file__)

train_labels, train_images = mnist.read_csv(os.path.join(dirname, '../data/mnist_train.csv'))
DATASET = mnist.DataSet(train_images, train_labels)
OUT = os.path.join(dirname, "../models/mnist")

batch_size = 128
num_steps = 1800
learning_rate = 0.01
start = time.time()

# input
x = tf.placeholder(tf.float32, [None, 784], "x")
y_ = tf.placeholder(tf.float32, [None, 10], "y")

# weight
W = tf.Variable(tf.zeros([784, 10]))
# bias
b = tf.Variable(tf.zeros([10]))
# test_data * W + b
y = tf.matmul(x, W) + b
sm = tf.nn.softmax(y, name="softmax")

# cross entropy (loss function)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_), name="loss")

# train step
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# evaluating the model
correct_prediction = tf.equal(tf.argmax(sm, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    # training
    for step in range(num_steps):
        batch_data, batch_labels = DATASET.next_batch(batch_size)
        feed_dict = {x: batch_data, y_: batch_labels}
        
        loss_out, ts_out, acc_out = session.run([loss, train_step, accuracy], feed_dict=feed_dict)  

    save_path = saver.save(session, OUT)

    with open(os.path.join(dirname, '../metrics/train.json'), 'w') as outfile:
        json.dump({ 
            "batch_size": batch_size, 
            "num_steps": num_steps, 
            "learning_rate": learning_rate,  
            "took" : (time.time() - start) / 1000 }, outfile)

