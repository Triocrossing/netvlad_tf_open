import numpy as np
import time
import matplotlib.pyplot as plt
import glob
import os
import cv2
import sys
# Much faster than the standard class
from fast_slic.avx2 import SlicAvx2
from PIL import Image

def showMat(mat, figName="", isGrid=True):
  if figName:
    plt.figure(figName,figsize=(15,10))
  else:
    plt.figure()
  plt.imshow(mat)

def genSP(imageDir, ext):
  
  imgPath = sorted(glob.glob(os.path.join(imageDir, '*.'+ext)))

  # prepare SP folder
  parentdir = os.path.dirname(imageDir)
  nameWtExt = os.path.splitext(os.path.basename(imageDir))[0]
  newFolderName = parentdir+"/" +nameWtExt+"_superpixelGen"
  if not os.path.exists(newFolderName):
    os.mkdir(newFolderName)

  for i in range(0, len(imgPath)):
    start = time.time()
    image = cv2.imread(imgPath[i])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) 
    slic = SlicAvx2(num_components=60,compactness=20,min_size_factor=0)
    assignment = slic.iterate(image) # Cluster Map
    fnameWtExt = os.path.splitext(os.path.basename(imgPath[i]))[0]
    
    # we transpose it here
    assignment = np.transpose(assignment)
    # save SP
    f = cv2.FileStorage(newFolderName+'/'+fnameWtExt+'.yml',flags=1)
    f.write(name='mat',val=assignment)
    f.release()

    showMat(assignment, "slic")
    plt.savefig(newFolderName+'/sp_'+fnameWtExt+'.jpg')
    plt.clf()

    end = time.time()
    print(f'image No. {i}, time per loop: {end - start:.2f} s,  remaining {(end - start)*(len(imgPath)-i)/60:.2f} mins')
    print ("\033[A\033[A")
    # print(f'time is : {end-start}')

    # viz
    # plt.plot("slic", assignment)
    # cv2.waitKey(0)
    # showMat(assignment, "slic")
    # showMat(image, "image")
    # plt.show()


def main(arg):

  imgFolder = arg[0]
  ext = arg[1]
  genSP(imgFolder, ext)
  exit()
  # dir to SP



if __name__ == "__main__":
    main(sys.argv[1:])