import glob
import numpy as np
import os
import sys

def main(arg):
  jpeg_paths = sorted(glob.glob(os.path.join(arg[0], '*.'+arg[1])))
  filename = (os.path.basename(os.path.dirname(arg[0]))+"_rear.txt")
  # print(jpeg_paths)
  with open(filename, "w") as f:
    for s in jpeg_paths:
        f.write(str(s) +"\n")

if __name__ == "__main__":
    main(sys.argv[1:])