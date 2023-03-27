# Env-emlp

mkdir /root/.mujoco && cp -r /root/yanjing-nfs-hdd/chengh-wang/BBQ/RPP/mujoco210 /root/.mujoco/mujoco210

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

apt install libgl1-mesa-dev

apt install patchelf

apt install libosmesa6-dev libgl1-mesa-glx libglfw3

pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install mujoco-py

pip install ml_collections

pip install tensorboardX

pip install flax

pip install gym

pip install d4rl

pip install gdown

pip install objax

pip install emlp

pip install olive-oil-ml

pip install distrax

pip install imageio[ffmpeg]

pip install imageio[pyav]

pip install wandb

pip install absl-py==0.12.0

pip install numpy



