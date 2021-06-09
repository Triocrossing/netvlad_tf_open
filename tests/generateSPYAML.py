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
import glob
import struct

from netvlad_tf.image_descriptor import ImageDescriptor
import netvlad_tf.net_from_mat as nfm
import netvlad_tf.precision_recall as pr

def showMat(mat, figName="", isGrid=True):
  if figName:
    plt.figure(figName,figsize=(15,10))
  else:
    plt.figure()
  plt.imshow(mat)
  plt.colorbar()
  plt.xticks(np.arange(0, float(np.shape(mat)[1])+1, 100.0),rotation='vertical')
  plt.xticks(fontsize=8)
  plt.yticks(np.arange(0, float(np.shape(mat)[0])+1, 100.0))
  plt.yticks(fontsize=8)
  if isGrid:
    plt.grid(color='r', linestyle='-', linewidth=1)

def bbox2(img):
  rows = np.any(img, axis=1)
  cols = np.any(img, axis=0)
  ymin, ymax = np.where(rows)[0][[0, -1]]
  xmin, xmax = np.where(cols)[0][[0, -1]]
  # print("ymin ymax, xmin xmax")
  return img[ymin:ymax+1, xmin:xmax+1], ymin, ymax+1, xmin, xmax+1


def createEnveloppedArrayBySP(spImg, img):
  # print(f'shape sp{np.shape(spImg)}')
  # print(f'shape img{np.shape(img)}')
  assert(np.shape(spImg)==np.shape(img))
  lenvImg = []                                            # list of envelopped images
  for spIdx in np.unique(spImg):                          # spIdx is the idx of clustering
    # print(f'spIdx: {spIdx}')
    _img = np.copy(img)                                   # copy to prevent changing original data
    _spImg = np.copy(spImg)                               # copy to prevent changing original data
    _spImg[spImg!=spIdx] = -1                             # set zero for non-idx entries
    _spImg[spImg==spIdx] = 0                              # set zero for non-idx entries
    _spImg=_spImg+1
    _spImg, ymin, ymaxp1, xmin, xmaxp1 = bbox2(_spImg)    # bounding box non zero regions
    _img = _img[ymin:ymaxp1, xmin:xmaxp1]
    _img = _img*_spImg
    # showMat(_img,"after bb")
    # plt.show()
    lenvImg.append(_img)
  
  print(f'found {len(lenvImg)} sp regions')
  return lenvImg
    
# a priori int
def readSPBin(fileNaf):
  # a=array
  with open(fileNaf, "rb") as f:
    # byte = f.read(1)
    # read the header
    rows     = struct.unpack('i', f.read(4))[0]
    cols     = struct.unpack('i', f.read(4))[0]
    type     = struct.unpack('i', f.read(4))[0]
    channels = struct.unpack('i', f.read(4))[0]
    
    print(f'image size of: {rows} x {cols}, of type {type} and channel {channels}')
    byte = f.read(4*rows*cols*channels)
    a = np.frombuffer(byte, dtype=np.int32)
    a = a.reshape(rows, cols)
    a = np.transpose(a)
    # print(a)
    # showMat(a)
    # plt.show()
    # cv2.imshow("a",a)
  return a

def describeSPNetvlad(sp, img, imd):
  lfeat = []
  lenv = createEnveloppedArrayBySP(sp, img)
  for subImg in lenv:
    feat = imd.describe(subImg, False)      # do not squeeze here
    lfeat = lfeat + list(feat)
  return lfeat

def describeSPNetvladAll(pathSP, pathImg, ext):
  tf.reset_default_graph()
  imd = ImageDescriptor(is_grayscale=True)
  # we search all bin
  spPath = sorted(glob.glob(os.path.join(pathSP, '*.bin')))
  imgPath = sorted(glob.glob(os.path.join(pathImg, '*.'+ext)))

  # print first five for verification
  print(imgPath[:5])
  lfeat = []
  use_dim = 4096
  
  parentdir = os.path.dirname(pathImg)
  nameWtExt = os.path.splitext(os.path.basename(pathImg))[0]
  newFolderName = parentdir+"/" +nameWtExt+"genYAML"
  if not os.path.exists(newFolderName):
    os.mkdir(newFolderName)

  for i in range(0, len(spPath)):
    sp = readSPBin(spPath[i])
    image = cv2.imread(imgPath[i], cv2.IMREAD_GRAYSCALE)
    lfeat = describeSPNetvlad(sp, image, imd)
    use_feats = np.array(lfeat)[:, :use_dim]
    # print(np.array(use_feats).shape)
    fnameWtExt = os.path.splitext(os.path.basename(imgPath[i]))[0]
    
    # save lfeat as result of one image
    f = cv2.FileStorage(newFolderName+'/'+fnameWtExt+'_NetVladfeat.yml',flags=1)
    f.write(name='mat',val=use_feats)
    f.release()
    
  

def unitest(arg):
  a = readSPBin(arg[0])
  image = cv2.imread(arg[1], cv2.IMREAD_GRAYSCALE)
  showMat(a)
  plt.show()
  exit()


def main(arg):
  # unitest(arg)
  # l1, l2 = readCsv(arg[0])
  if tf.test.is_gpu_available():
    print("GPU is available")
  else:
    print("GPU is NOT available")

  imgFolder = arg[0]
  ext = arg[1]
  spFolder = arg[2]
  describeSPNetvladAll(spFolder,imgFolder, ext)
  exit()
  # dir to SP


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