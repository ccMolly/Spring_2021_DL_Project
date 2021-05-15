FROM ubuntu:18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
RUN mkdir /app/templates
RUN mkdir /app/user_imgs
RUN mkdir /app/models
RUN mkdir /app/vgg
RUN mkdir /app/resnet
RUN mkdir /app/vgg_pruned
RUN mkdir /app/resnet_pruned
RUN mkdir /app/vgg_trained_models
RUN mkdir /app/resnet_trained_models
WORKDIR /app

ADD code/* /app/
ADD code/templates /app/templates
ADD code/user_imgs /app/user_imgs
ADD code/models/ /app/models/
ADD code/vgg/ /app/vgg/
ADD code/resnet/ /app/resnet/
ADD code/vgg_pruned/ /app/vgg_pruned/
ADD code/resnet_pruned/ /app/resnet_pruned/
ADD code/vgg_trained_models/ /app/vgg_trained_models/
ADD code/resnet_trained_models/ /app/resnet_trained_models/

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

# No CUDA-specific steps
ENV FLASK_APP=main.py
ENV NO_CUDA=1
ENV PORT 5000
EXPOSE $PORT

RUN conda install -y -c pytorch \
    cpuonly \
    "pytorch=1.5.0=py3.8_cpu_0" \
    "torchvision=0.6.0=py38_cpu" \
    "flask" \
 && conda clean -ya

# Set the default command to python3
CMD ["python3", "main.py"]
