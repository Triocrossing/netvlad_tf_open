import cv2
import glob
import numpy as np
import os
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() 

import netvlad_tf.nets as nets


class ImageDescriptor(object):

    def __init__(self, is_grayscale=False):
        self.is_grayscale = is_grayscale
        if is_grayscale:
            self.tf_batch = tf.placeholder(
                    dtype=tf.float32, shape=[None, None, None, 1])
        else:
            self.tf_batch = tf.placeholder(
                    dtype=tf.float32, shape=[None, None, None, 3])
        self.net_out = nets.vgg16NetvladPca(self.tf_batch)
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, nets.defaultCheckpoint())

    def describeAllFilesInPath(self, path, batch_size, ext='png', verbose=False):
        ''' returns a list of descriptors '''
        jpeg_paths = sorted(glob.glob(os.path.join(path, '*.'+ext)))
        print(jpeg_paths)
        descs = []
        # for batch_offset in range(0, len(jpeg_paths), batch_size):
        for batch_offset in range(0, 4, batch_size):
            images = []
            # creating batch
            for i in range(batch_offset, batch_offset + batch_size):
                if i == len(jpeg_paths):
                    break
                if verbose:
                    print('%d/%d' % (i, len(jpeg_paths)))
                if self.is_grayscale:
                    image = cv2.imread(jpeg_paths[i], cv2.IMREAD_GRAYSCALE)
                    images.append(np.expand_dims(
                            np.expand_dims(image, axis=0), axis=-1))
                else:
                    image = cv2.imread(jpeg_paths[i])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(np.expand_dims(image, axis=0))
            batch = np.concatenate(images, 0)
            descs = descs + list(self.sess.run(
                    self.net_out, feed_dict={self.tf_batch: batch}))
        return descs

    def describe(self, image, isSqueeze=True):
        if self.is_grayscale:
            batch = np.expand_dims(np.expand_dims(image, axis=0), axis=-1)
        else:
            batch = np.expand_dims(image, axis=0)
        if isSqueeze:
          return self.sess.run(self.net_out, feed_dict={self.tf_batch: batch}).squeeze()
        else:
          return self.sess.run(self.net_out, feed_dict={self.tf_batch: batch})

