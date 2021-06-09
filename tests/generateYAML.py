# This takes a lot of time to run, so not written as unit test.

import netvlad_tf.nets as nets

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import scipy.spatial.distance as scid
import scipy.signal as scisig
import tensorflow as tf
import os
# import tensorflow.compat.v1 as tf
import time
import unittest
import sys

from netvlad_tf.image_descriptor import ImageDescriptor
import netvlad_tf.net_from_mat as nfm
import netvlad_tf.precision_recall as pr

def main(arg):
  # l1, l2 = readCsv(arg[0])
  if tf.test.is_gpu_available():
    print("GPU is available")
  else:
      print("GPU is NOT available")

  folderdir = arg[0]
  ext = arg[1]

  parentdir = os.path.dirname(arg[0])
  nameWtExt = os.path.splitext(os.path.basename(arg[0]))[0]
  # Init
  use_dim = 4096

  # Get KITTI feats
  tf.reset_default_graph()
  imd = ImageDescriptor(is_grayscale=True)
  feats = imd.describeAllFilesInPath(folderdir, 4, ext, verbose=True)
  # print (feats)
  # cv2.Save("test.xml",cv2.fromarray(feats))
  print(np.array(feats).shape)
  use_feats = np.array(feats)[:, :use_dim]
  f = cv2.FileStorage(parentdir+'/'+nameWtExt+'_NetVladfeat.yml',flags=1)
  f.write(name='mat',val=use_feats)
  f.release()
  # exit()  
  # feats = imd.describeAllFilesInPath('/Users/triocrossing/INRIA/NOSAVE/Datasets/KITTI/00/image_2/'


if __name__ == "__main__":
    main(sys.argv[1:])