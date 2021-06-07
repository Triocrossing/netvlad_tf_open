import cv2
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
# import pandas as pd
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import sys
import re
import re

def saveMat(mat, filename):
  f = cv2.FileStorage(filename,flags=1)
  f.write(name='mat',val=mat)
  f.release()

def getPosFromName(df, filename):
  Pos = np.zeros((3,1))
  # idxMatch = df.loc[df[0] == filename]
  # print(f'searching for {filename}')
  subdf = df[df[0].str.match(filename)]
  # subdf = df[df[0] == filename]
  if(subdf.empty):
    return Pos, False

  # print(f'subdf {subdf}')
  Pos[0]=subdf[4]
  Pos[1]=subdf[8]
  Pos[2]=subdf[12]
  return Pos, True

def readGTtxt(filename):
  df = pd.read_csv(filename, delimiter=' ', header=None)
  # image_name R00 R01 R02 C0 R10 R11 R12 C1 R20 R21 R22 C2 0 0 0 1
  print(df.head(5))
  return df
  # 0 for image name

  # 4 8 12 for C0C1C2 translational information

def readList(filename):
  list = []
  with open(filename, "r") as f:
    for line in f:
      fullname = str(line.strip())
      filename = path.basename(fullname)
      filename = path.basename(path.dirname(fullname)) +"/"+ filename
      filename = path.basename(path.dirname(path.dirname(fullname))) +"/"+  filename
      # print(filename)
      list.append(filename)
  return list

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

def loadGTMat(arg, mName="mat"):
  print(arg)
  readerm = cv2.FileStorage(arg, cv2.FILE_STORAGE_READ)
  mFeat = readerm.getNode(mName).mat()
  return mFeat

def BFbestSearch(refM, qryM, lR, lQ, dfR, dfQ):
  rowlenRef = len(refM)
  rowlenQry = len(qryM)
  print(f'ref image : {rowlenRef}, qry image {rowlenQry}')
  mcov = np.zeros((rowlenRef, rowlenQry))
  mdist = np.zeros_like(mcov)

  ctr = 0
  rowidx=0
  for idx_row_q in range(0, rowlenQry):
    for idx_row_r in range(0, rowlenRef):
      # print(refM[idx_row_r,:])
      mcov[idx_row_r,idx_row_q] = np.linalg.norm(refM[idx_row_r,:] - qryM[idx_row_q,:]) 

      # dist
      PosR, isFoundR = getPosFromName(dfR, lR[idx_row_r])
      PosQ, isFoundQ = getPosFromName(dfQ, lQ[idx_row_q])
      if(isFoundR and isFoundQ):
        mdist[idx_row_r,idx_row_q] = np.linalg.norm(PosR-PosQ)
      else:
        mdist[idx_row_r,idx_row_q] = -1 
    print(f'precessing query image No. {idx_row_q}')
    #
  return mcov, mdist

def analysis(mcov, mdist):
  lDist = []
  minXIdxForY = np.argmin(mcov, axis=1)
  for m in range(0,len(minXIdxForY)):
    maxn = minXIdxForY[m]
    if(mdist[m,maxn]>0):
      lDist.append(mdist[m,maxn])
  plt.plot(lDist)
  plt.plot(sorted(lDist))
  return lDist

def main(arg):
  referenceMat = loadGTMat(arg[0])
  queryMat = loadGTMat(arg[1])
  refList = readList(arg[2])
  queryList = readList(arg[3])
  dfR = readGTtxt(arg[4])
  dfQ = readGTtxt(arg[5])
  # print(refList[:5])
  # gtList = readList(arg[4])
  resName = str(path.basename(arg[2]))[:-4] + "_vs_"+str(path.basename(arg[3]))[:-4] 
  print(resName)
  # exit()

  mcov, mdist = BFbestSearch(referenceMat, queryMat, refList, queryList, dfR, dfQ)

  lDist = analysis(mcov, mdist)
  saveMat(mcov, 'mcov'+resName)
  saveMat(mdist, 'mdist'+resName)
  showMat(mcov)
  showMat(mdist)
  plt.show()

def mainAlt(arg):
  mcov = loadGTMat(arg[0])
  mdist = loadGTMat(arg[1])

  lDist = analysis(mcov, mdist)

  showMat(mcov)
  showMat(mdist)
  plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
    # mainAlt(sys.argv[1:])
