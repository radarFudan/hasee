FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
USER root:root

ARG IMAGE_NAME=None
ARG BUILD_NUMBER=None
ARG DEBIAN_FRONTEND=noninteractive

ENV com.nvidia.cuda.version $CUDA_VERSION
ENV com.nvidia.volumes.needed nvidia_driver
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV NCCL_DEBUG=INFO
ENV HOROVOD_GPU_ALLREDUCE=NCCL

# Install Common Dependencies
RUN apt-get update && \
    # Others
    apt-get install -y libksba8 \
    openssl \
    libaio-dev \
    git \
    wget

# RUN pip install deepspeed==0.10.2
# RUN pip install --index-url https://download.pytorch.org/whl/nightly/cu121 --pre 'torch>=2.1.0dev'
RUN pip install ninja -U
RUN pip uninstall -y flash-attn
# set maximum workers
ENV MAX_JOBS=4
# RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
RUN git clone https://github.com/Dao-AILab/flash-attention
RUN cd flash-attention && \
    python setup.py install && \
    cd csrc/rotary && pip install . && \
    cd ../layer_norm && pip install . && \
    cd ../xentropy && pip install . && \ 
    cd ../.. && rm -rf flash-attention

RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
# RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

RUN pip install bitsandbytes==0.40.0 transformers==4.31.0 peft==0.4.0 accelerate==0.21.0 einops==0.6.1 evaluate==0.4.0 scikit-learn==1.2.2 sentencepiece==0.1.99 wandb==0.15.3 tokenizers
RUN pip install git+https://github.com/Lightning-AI/lightning@800b87eb464bda8defa9425bb0b76651c5c5175b jsonargparse[signatures] pandas pyarrow tokenizers wandb zstd
RUN pip uninstall -y transformer_engine
RUN pip install lightning[app]

RUN pip3 install jax==0.4.13
# RUN pip3 install -U "jaxlib==0.4.13+cuda12.cudnn89" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install -U "jaxlib==0.4.13+cuda11.cudnn86" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip3 install pipreqs
RUN pip3 install omegaconf


# Install dependencies
COPY apt_install.txt .
RUN apt-get update
RUN apt-get install -y `cat apt_install.txt`
RUN npm i -g @bazel/bazelisk
RUN npm i -g @bazel/buildifier
# Config pip
RUN python3 -m pip config set global.index-url http://pypi.ai.seacloud.garenanow.com/root/dev
RUN python3 -m pip config set global.trusted-host pypi.ai.seacloud.garenanow.com
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip, install py libs
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt --upgrade

COPY tools /tools
