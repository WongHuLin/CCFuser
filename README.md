# Harnessing inter-gpu shared memory for seamless moe communication-computation fusion (PPoPP'25)

This repo is for PPoPP 2025 artifacts evaluation.

### Pulling docker images and booting

export CCFuser_HOME=<ccfuser_home>

docker pull nvcr.io/nvidia/pytorch:22.07-py3

nvidia-docker run --privileged  -it -d --shm-size=10g --ulimit memlock=-1 -v $CCFuser_HOME:/workspace  --name ppopp_ae nvcr.io/nvidia/pytorch:22.07-py3 /bin/bash

### install conda
```bash
export ROOT_PATH=<root_path>

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
conda create --name ccfuser python=3.10
conda activate ccfuser
```

### install nccl
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list
rm /etc/apt/sources.list.d/cuda.list 
apt-get update

apt install libnccl2=2.14.3-1+cuda11.7 libnccl-dev=2.14.3-1+cuda11.7 
```

### install openmpi
```bash
export OPENMPI_HOME=$ROOT_PATH/local/openmpi
cd $ROOT_PATH/local
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
tar -zxvf openmpi-4.1.1.tar.gz
autogen.pl --force
./configure  --enable-mpi-cxx --prefix=$OPENMPI_HOME --with-cuda=/usr/local/cuda 
make all
make install
```

### install nvshmem
```bash
cd $ROOT_PATH/local/nvshmem_src_2.9.0-2
export NCCL_IB_DISABLE=0
export CUDA_HOME=/usr/local/cuda-11.7
export NVSHMEM_MPI_SUPPORT=1
export MPI_HOME=$ROOT_PATH/local/openmpi
export NVSHMEM_PREFIX=$ROOT_PATH/local/nvshmem
mkdir build
cd build
cmake ..
make -j8 install
```

### install python dependency
```bash
pip install -r $ROOT_PATH/mix_moe/requirements.txt
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### install tutel and fastmoe
```bash
git clone https://github.com/microsoft/tutel.git
cd tutel
python setup.py install

git clone https://github.com/laekov/fastmoe.git
cd fastmoe
python setup.py install
```

### fig10
```bash
cd $ROOT_PATH/mix_moe/ae
bash $ROOT_PATH/mix_moe/ae/fig10_data.sh
python fig10_plot.py 
```

### fig11
```bash
cd $ROOT_PATH/mix_moe/ae
bash $ROOT_PATH/mix_moe/ae/fig11_data.sh
python fig11_plot.py 
```

### fig12
```bash
cd $ROOT_PATH/mix_moe/ae
bash $ROOT_PATH/mix_moe/ae/fig12_data.sh
python fig12_plot.py 
```

###  Citation
If you use CCFuser in your research, please consider citing our paper:
```bash
@inproceedings{wang2025harnessing,
  title={Harnessing inter-gpu shared memory for seamless moe communication-computation fusion},
  author={Wang, Hulin and Xia, Yaqi and Yang, Donglin and Zhou, Xiaobo and Cheng, Dazhao},
  booktitle={Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages={170--182},
  year={2025}
}

```

