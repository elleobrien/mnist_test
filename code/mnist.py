import os
import sys
import gzip
import shutil

import numpy as np

import wget

def download(uri, path):
  wget.download(uri, path)

def unzip(path):
    input = gzip.GzipFile(path, 'rb')
    s = input.read()
    input.close()

    output = open(path.replace('.gz', ''), 'wb')
    output.write(s)
    output.close()

def get_images(imgf, n):
    f = open(imgf, "rb")
    f.read(16)
    images = []

    for i in range(n):
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    return images

def get_labels(labelf, n):
    l = open(labelf, "rb")
    l.read(8)
    labels = []
    for i in range(n):
        labels.append(ord(l.read(1)))
        
    return labels

def output_csv(folder, images, labels, prefix):
    if not os.path.exists(folder):
        os.mkdir(folder)

    o = open(os.path.join(folder, "mnist_%s.csv"%prefix), "w")
    for i in range(len(images)):
        o.write(",".join(str(x) for x in [labels[i]] + images[i]) + "\n")
    o.close()

def process(folder, imgf, labelf, prefix, n):
    images = get_images(os.path.join(folder, imgf), n)
    labels = get_labels(os.path.join(folder, labelf), n)
    output_csv(folder, images, labels, prefix)
    
def read_csv(path):
    labels = []
    imgs = []

    with open(path) as f:
        for i, line in enumerate(f): 
            data = line.split(',')  

            label = data[0]
            label_one_hot = np.zeros(10)
            label_one_hot[int(label)] = 1
            labels.append(label_one_hot)

            img = np.array(data[1:])
            img = img.astype(np.float32)
            img = np.multiply(img, 1.0 / 255.0)
            imgs.append(img)
    
    return (np.asarray(labels), np.asarray(imgs))


class DataSet(object):
  def __init__(self, images, labels):   
    self.num_examples = images.shape[0]
    self.images = images
    self.labels = labels
    self.epochs_completed = 0
    self.index_in_epoch = 0

  def next_batch(self, batch_size):
    start = self.index_in_epoch
    self.index_in_epoch += batch_size

    if self.index_in_epoch > self.num_examples:
      self.epochs_completed += 1
      
      # Shuffle the data
      perm = np.arange(self.num_examples)
      np.random.shuffle(perm)
      self.images = self.images[perm]
      self.labels = self.labels[perm]

      # Start next epoch
      start = 0
      self.index_in_epoch = batch_size
      assert batch_size <= self.num_examples

    end = self.index_in_epoch
    return self.images[start:end], self.labels[start:end]


if __name__== "__main__":
    if len(sys.argv) < 2:
        print('folder is missing. Run command with folder path.')
        exit(1)

    out_folder = sys.argv[1]

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    else:
        print('folder ' + out_folder + ' already exists! Delete it with all its content in order to prepare it')
        exit(1)

    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz', 
        't10k-images-idx3-ubyte.gz', 
        't10k-labels-idx1-ubyte.gz' ]

    for fil in files:
        path = os.path.join(out_folder, fil)
        download(SOURCE_URL + fil, out_folder)
        unzip(path)

    process(out_folder, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 'train', 60000)
    process(out_folder, "t10k-images-idx3-ubyte",  "t10k-labels-idx1-ubyte", 'test', 10000)

    for filename in files:
        path = os.path.join(out_folder, filename)
        os.remove(path)
        os.remove(path.replace('.gz', ''))
