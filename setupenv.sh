source /macierz/home/s165115/.bashrc
conda activate mgr
export PATH=${CUDA_HOME}/bin:$PATH
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/LEGO/agent/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.3/bin/:$PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.3
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices
