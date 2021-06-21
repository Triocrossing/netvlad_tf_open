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
from fast_slic.avx2 import SlicAvx2

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

def showMatSP(mat, mat2, figName="", isGrid=True):
  if figName:
    plt.figure(figName,figsize=(15,10))
  else:
    plt.figure()
  plt.imshow(mat2,  cmap='gray', vmin=0, vmax=255)
  plt.imshow(mat, alpha=0.9)

def bbox2(img):
  rows = np.any(img, axis=1)
  cols = np.any(img, axis=0)
  ymin, ymax = np.where(rows)[0][[0, -1]]
  xmin, xmax = np.where(cols)[0][[0, -1]]
  # print("ymin ymax, xmin xmax")
  return img[ymin:ymax+1, xmin:xmax+1], ymin, ymax+1, xmin, xmax+1
  
def createEnveloppedArrayBySP(spImg, _img, isZeroPadding=False):
  img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
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
    if(isZeroPadding):
      _img = _img*_spImg
    # print(f'shape of _img {np.shape(_img)} and shape of _spImg {np.shape(_spImg)}')
    # showMat(_img,"after zeroPading")
    # plt.show()
    # print)
    lenvImg.append(_img)
    # exit()
  
  # print(f'found {len(lenvImg)} sp regions')
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
    
    # print(f'image size of: {rows} x {cols}, of type {type} and channel {channels}')
    byte = f.read(4*rows*cols*channels)
    a = np.frombuffer(byte, dtype=np.int32)
    a = a.reshape(rows, cols)
    a = np.transpose(a)
    # print(a)
    # showMat(a)
    # plt.show()
    # cv2.imshow("a",a)
  return a

# typecv float 5
# typecv int 4
def writeSPBin(sp, fileSP, cvtype=4, isTranspose=False):
  if(isTranspose):
    sp=np.transpose(sp)
  num_rows, num_cols = sp.shape
  if cvtype==4:
    sp = sp.astype('int32')
  if cvtype==5:
    sp = sp.astype('float32')
  with open(fileSP, "wb") as f:
    # write the header
    rows     = struct.pack('i', num_rows)
    f.write(rows)
    cols     = struct.pack('i', num_cols)
    f.write(cols)
    sptype   = struct.pack('i', cvtype)
    f.write(sptype)
    channels = struct.pack('i', 1)
    f.write(channels)
    
    # print(f'image size of: {rows} x {cols}, of type {type} and channel {channels}')
    spbytes = sp.tobytes()
    f.write(spbytes)

def describeSPNetvlad(sp, img, imd):
  lfeat = []
  lenv = createEnveloppedArrayBySP(sp, img)
  for subImg in lenv:
    # print(f'shape of _img {np.shape(subImg)} and shape of _spImg {np.shape(subImg)}')
    feat = imd.describe(subImg, False)      # do not squeeze here
    lfeat = lfeat + list(feat)
  return lfeat

def describeSPNetvladAll(pathImg, ext, nSP):
  tf.reset_default_graph()
  imd = ImageDescriptor(is_grayscale=True)
  # we search all bin
  # spPath = sorted(glob.glob(os.path.join(pathSP, '*.bin')))
  imgPath = sorted(glob.glob(os.path.join(pathImg, '*.'+ext)))

  # print first five for verification
  print(imgPath[:5])
  lfeat = []
  use_dim = 4096

  parentdir = os.path.dirname(pathImg)
  nameWtExt = os.path.splitext(os.path.basename(pathImg))[0]
  newFolderName = parentdir+"/nSP_"+str(nSP)+"_BiGNetvlad"+nameWtExt

  newFolderNameYML = parentdir+"/nSP_"+str(nSP)+"_yml_"+nameWtExt

  if not os.path.exists(newFolderNameYML):
    os.mkdir(newFolderNameYML)

  newFolderNameCLR = parentdir+"/nSP_"+str(nSP)+"_clr_"+nameWtExt
  if not os.path.exists(newFolderNameCLR):
    os.mkdir(newFolderNameCLR)

  if not os.path.exists(newFolderName):
    os.mkdir(newFolderName)
  # assert(len(spPath)==len(imgPath))
  print("...")
  for i in range(0, len(imgPath)):
    print ("\033[A\033[A")
    fnameWtExt = os.path.splitext(os.path.basename(imgPath[i]))[0]
    if os.path.exists(newFolderNameYML+'/'+fnameWtExt+'.bin'):
      print ("file already exist ...")
      continue

    start = time.time()

    image = cv2.imread(imgPath[i])
    slic = SlicAvx2(num_components=nSP,compactness=30,min_size_factor=0.5)
    sp = slic.iterate(image) # Cluster Map

    # sp = readSPBin(pathSP + "/" + os.path.basename(imgPath[i])[:-4]+".bin")
    # showMat(sp)
    # plt.show()
    lfeat = describeSPNetvlad(sp, image, imd)
    use_feats = np.array(lfeat)[:, :use_dim]
    # print(np.array(use_feats).shape)
    
    # save SP
    # f = cv2.FileStorage(newFolderNameYML+'/'+fnameWtExt+'.yml',flags=1)
    # f.write(name='mat',val=sp)
    # f.release()
    writeSPBin(sp, newFolderNameYML+'/'+fnameWtExt+'.bin',4,True)

    showMatSP(sp, image, "slic")
    plt.savefig(newFolderNameCLR+'/sp_'+fnameWtExt+'.jpg')
    plt.clf()

    # save lfeat as result of one image
    # f = cv2.FileStorage(newFolderName+'/'+fnameWtExt+'.yml',flags=1)
    # f.write(name='mat',val=use_feats)
    # f.release()

    # double and non-tranpose
    writeSPBin(use_feats, newFolderName+'/'+fnameWtExt+'.bin',5,False)

    end = time.time()
    print(fnameWtExt)
    print(f'image No. {i}, time per loop: {end - start:.2f} s, remaining {(end - start)*(len(imgPath)-i)/60:.2f} mins')
    # print ("\033[A\033[A")
    
  

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
  nSP = int(arg[2])
  describeSPNetvladAll(imgFolder, ext, nSP)
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