#!/bin/bash
#patch to be aware of "module" inside a job
#OAR -l {host ='igrida-abacus15.irisa.fr'}/gpu_device=1,walltime=24:00:0
#OAR -O /srv/tempdd/xwang/logNetvladSPYAML.%jobid%.output
#OAR -E /srv/tempdd/xwang/logNetvladSPYAML.%jobid%.error

echo "Start init ..."

. /etc/profile.d/modules.sh

module load miniconda
# module load spack/cuda/8.0.61/gcc-8.3.0-gpuxnlq
module load spack/cudnn/7.2.1.38-9.0/gcc-8.3.0-nlq7dqt

# module load spack/cudann/6.
# module load spack/py-numpy/1.19.2/gcc-8.3.0-wrsq6vj
# module load spack/py-setuptools/41.4.0/gcc-8.3.0-7wtfm6i

# pip install tensorflow-gpu==1.9
# pip3 install numpy==1.19.5
# module load numpy
# source ~/anaconda3/etc/profile.d/conda.sh

conda init bash
source ~/.bashrc
source /soft/igrida/miniconda/miniconda-3/etc/profile.d/conda.sh
conda env list
# eval "$(conda shell.bash hook)"
# conda activate netvlad-tf
eval "$(conda shell.bash hook)"
# source /soft/igrida/miniconda/miniconda-3/bin/activate netvlad-tf
source activate netvlad-tf
conda env list
# python --version
# condaNetvlad
# conda list
# pip install tensorflow-gpu==1.9

cd ~/xwang/netvlad_tf_open

# export PYTHONPATH="/udd/xwang/xwang/netvlad_tf_open/python":$PYTHONPATH
export PYTHONPATH="python"
echo $PYTHONPATH

conda env list
echo "Start computing ..."

source activate netvlad-tf
# python tests/test.py
# python tests/generateSPYAML.py /srv/tempdd/xwang/oldRobotCarSeason/reference-left/left jpg /srv/tempdd/xwang/oldRobotCarSeason/reference-left/Bin_yml
python tests/generateSPYAMLInverse.py /srv/tempdd/xwang/oldRobotCarSeason/overcast-reference/rear jpg 70

cd /srv/tempdd/xwang/oldRobotCarSeason/overcast-reference
mkdir folder_nSP_70 
mv nSP_70* folder_nSP_70
zip -r nSP_70.zip folder_nSP_70

# conda env list

# conda activate xwang
# module load spack/cuda/10.0.130/gcc-8.3.0-kywfj57
# module load spack/cudnn/7.6.5.32-10.1-linux-x64/gcc-8.3.0-qryyh6p

# conda env remove --name xwang 
# conda env create -n xwang python==3.6
# pip install matplotlib
# pip install tensorflow-gpu==2.0.0
# pip install opencv-python
# pip install tqdm
# pip uninstall h5py
# pip install h5py==2.9.0

# cuda 8 for 1.9
# cudnn 7.1