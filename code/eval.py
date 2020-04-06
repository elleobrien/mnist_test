import os
import json
import tensorflow.compat.v1 as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.disable_v2_behavior()

import mnist

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + ',\n')
    

dirname = os.path.dirname(__file__)

LABELS, IMAGES = mnist.read_csv(os.path.join(dirname, '../data/mnist_test.csv'))

META = os.path.join(dirname, '../models/mnist.meta')
MODELS = os.path.join(dirname, '../models/')

init = tf.global_variables_initializer()
with tf.Session() as sess:

    # Load trained model
    saver = tf.train.import_meta_graph(META)
    saver.restore(sess, tf.train.latest_checkpoint(MODELS))

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    softmax = graph.get_tensor_by_name("softmax:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    feed_dict = { x: IMAGES, y: LABELS }

    pred = sess.run([softmax, accuracy], feed_dict=feed_dict)
    prediction = tf.argmax(pred[0], axis=1, output_type=tf.int32)
    labels = tf.argmax(LABELS, axis=1, output_type=tf.int32)
    vals = tf.stack([labels,prediction]).eval().transpose()
    cfm= tf.math.confusion_matrix(
    labels, prediction, num_classes=10).eval()



for i in range(0,cfm.shape[0]):
    for j in range(0,cfm.shape[1]):
        cell_data = {
            '@experiment':'master',
            'label':str(i),
            'prediction':str(j),
            'count':int(cfm[i,j]),
        }
        dump_jsonl([cell_data], os.path.join(dirname, '../metrics/confusions.jsonl'), append=True)


# Write to file
with open(os.path.join(dirname, '../metrics/eval.json'), 'w') as outfile:
    json.dump({ "accuracy" : pred[1].item() }, outfile)
