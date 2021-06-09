import glob
import numpy as np
import os
import sys
import pandas as pd
import shutil as st

def readCsv(filename):
  df = pd.read_csv(filename, delimiter=',', header=None)
  print(df.head(5))
  list1 = df[0].tolist()
  list2 = df[1].tolist()
  return list1, list2

def main(arg):
  l1, l2 = readCsv(arg[0])
  # fileFolder1 = arg[1]
  fileFolder1 = "/Volumes/Xi_SSD_1To/Cross_Seasonal_Dataset/RobotCar-Seasons/" 
  fileFolder2 = "/Volumes/Xi_SSD_1To/Cross_Seasonal_Dataset/RobotCar-Seasons/" 
  # fileFolder2 = arg[2]

  foldername = os.path.dirname(arg[0])+"/matchRes"
  
  if not os.path.exists(foldername):
    os.mkdir(foldername)
  
  for idx in range(0, len(l1)):
    print(f'copying img {idx}')
    if not os.path.exists(foldername+"/"+str(idx)):
      os.mkdir(foldername+"/"+str(idx))
    baseFileName1 = os.path.basename(l1[idx])
    baseFileName2 = os.path.basename(l2[idx])
    st.copyfile(fileFolder1+"/"+l1[idx], foldername+"/"+str(idx)+"/"+baseFileName1)
    st.copyfile(fileFolder2+"/"+l2[idx], foldername+"/"+str(idx)+"/"+baseFileName2)

if __name__ == "__main__":
    main(sys.argv[1:])