import cv2
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
# import pandas as pd
import numpy as np
import os.path as path
import os
import matplotlib.pyplot as plt
import sys
import re
import re
import time

def saveList(list, name):
  with open(name, "w") as f:
    for idx1 in range(0, len(list)):
        f.write(str(list[idx1]) +"\n")

def saveMatchedList(list1, list2, name):
  with open(name, "w") as f:
    for idx1 in range(0, len(list1)):
        f.write(str(list1[idx1]) +"," + str(list2[idx1])  +"\n")

def saveMat(mat, filename):
  f = cv2.FileStorage(filename,flags=1)
  f.write(name='mat',val=mat)
  f.release()

def getPosFromName(df, filename, lName):
  Pos = np.zeros((3,1))
  # idxMatch = df.loc[df[0] == filename]
  # print(f'searching for {filename}')
  # subdf = df[df[0].loc(str(filename))]
  # subdf = df[df[0]==filename]
  # subdf = df[df[0].str.match(filename)]
  idx = next((i for i, v in enumerate(lName) if (filename==v)), -1)
  # print(f'found i: {idx}')
  if(idx==-1):
    return Pos, False
  subdf = df.iloc[idx]


  # subdf = df[df[0] == filename]
  if(subdf.empty):
    return Pos, False

  # print(f'subdf {subdf}')
  Pos[0]=subdf[4]
  Pos[1]=subdf[8]
  Pos[2]=subdf[12]
  # df = df.drop(df.index[subdf.index])
  return Pos, True

def getPosFromNameR(df, filename, lName):
  Pos = np.zeros((3,1))
  # idxMatch = df.loc[df[0] == filename]
  # print(f'searching for {filename}')
  # subdf = df[df[0].loc(str(filename))]
  # print(df.head(5))
  # print(filename)
  # subdf = df[df[0]==filename]
  # subdf = df[df[0].str.match(filename)]
  idx = next((i for i, v in enumerate(lName) if (filename==v)), -1)
  # print(f'found i: {idx}')
  if(idx==-1):
    return Pos, False
  subdf = df.iloc[idx]

  # subdf = df[df[0] == filename]


  # print(f'subdf {subdf}')
  Pos[0]=subdf[6]
  Pos[1]=subdf[7]
  Pos[2]=subdf[8]
  # df = df.drop(df.index[subdf.index])
  return Pos, True

def readGTtxt(filename):
  df = pd.read_csv(filename, delimiter=' ', header=None)
  df[0] = df[0].astype(str)
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
  listNameR = dfR[0].tolist()
  listNameQ = dfQ[0].tolist()
  for idx_row_q in range(0, rowlenQry):
    start = time.time()
    for idx_row_r in range(0, rowlenRef):
      # print(refM[idx_row_r,:])
      mcov[idx_row_r,idx_row_q] = np.linalg.norm(refM[idx_row_r,:] - qryM[idx_row_q,:]) 

      # dist
      PosR, isFoundR = getPosFromNameR(dfR, lR[idx_row_r], listNameR)
      PosQ, isFoundQ = getPosFromName(dfQ, lQ[idx_row_q], listNameQ)
      if(isFoundR and isFoundQ):
        mdist[idx_row_r,idx_row_q] = np.linalg.norm(PosR-PosQ)
      else:
        mdist[idx_row_r,idx_row_q] = -1 
    # print(f'precessing query ')
    # print("hello")
    end = time.time()
    print(f'image No. {idx_row_q}, time per loop: {end - start:.2f} s,  remaining {(end - start)*(rowlenQry-idx_row_q)/60:.2f} mins')
    print ("\033[A\033[A")

    #
  return mcov, mdist

def analysis(mcov, mdist, lR, lQ):
  lDist = []
  lMatchNameR = []
  lMatchNameQ = []
  minnIdxForm = np.argmin(mcov, axis=0)
  for n in range(0,len(minnIdxForm)):
    maxm = minnIdxForm[n]
    if(mdist[maxm,n]>0):
      lDist.append(mdist[maxm,n])
      lMatchNameR.append(lR[maxm])
      lMatchNameQ.append(lQ[n])
  return lDist, lMatchNameR, lMatchNameQ

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
  print(dfR[0].head(5))
  # print("dfR[0].iloc[0]")
  # print(dfR[0].iloc[0])
  # fname = 'overcast-reference/rear/1417176458999821.jpg'
  # print("searching")
  # subdf = dfR[dfR[0]==fname]
  # print(subdf)
  # print("dropping")
  # print(subdf.index)
  # dfR = dfR.drop(dfR.index[subdf.index])
  # print("after dropping")
  # print(dfR[0].head(5))


  # # print("dfR[0].loc[0]")
  # # print(dfR[0].loc[0])
  # exit()
  if not os.path.exists(resName):
    os.mkdir(resName)
  mcov, mdist = BFbestSearch(referenceMat, queryMat, refList, queryList, dfR, dfQ)

  # lDist = analysis(mcov, mdist)
  lDist, lMatchNameR, lMatchNameQ = analysis(mcov, mdist, refList, queryList)
  saveMatchedList(lMatchNameR, lMatchNameQ, resName+'/mtcList_'+resName+".csv")
  saveMat(mcov, resName+'/mcov_'+resName)
  saveMat(mdist, resName+'/mdist_'+resName)

  showMat(mcov)
  plt.savefig(resName+'/mcov.png')
  plt.clf()
  showMat(mdist)
  plt.savefig(resName+'/mdist.png')
  plt.clf()
  plt.plot(lDist)
  plt.plot(sorted(lDist))
  plt.savefig(resName+'/lDist.png')
  
  saveList(lDist, resName+'/lDist_'+resName+".csv")

def mainAlt(arg):
  mcov = loadGTMat(arg[0])
  mdist = loadGTMat(arg[1])
  refList = readList(arg[2])
  queryList = readList(arg[3])
  resName = str(path.basename(arg[2]))[:-4] + "_vs_"+str(path.basename(arg[3]))[:-4] 

  if not os.path.exists(resName):
    os.mkdir(resName)

  lDist, lMatchNameR, lMatchNameQ = analysis(mcov, mdist, refList, queryList)
  saveMatchedList(lMatchNameR, lMatchNameQ, resName+'/mtcList_'+resName+".csv")
  showMat(mcov)
  plt.savefig(resName+'/mcov.png')
  plt.clf()
  showMat(mdist)
  plt.savefig(resName+'/mdist.png')
  plt.clf()
  plt.plot(lDist)
  plt.plot(sorted(lDist))
  plt.savefig(resName+'/lDist.png')


if __name__ == "__main__":
    main(sys.argv[1:])
    # mainAlt(sys.argv[1:])
