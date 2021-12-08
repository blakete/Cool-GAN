FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 git -y

#Setup User
ARG USERNAME=blake
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $USERNAME
RUN useradd -l -m -u $UID -g $GID -o -s /bin/bash $USERNAME
WORKDIR /src
RUN chown -R $USERNAME:$USERNAME /src
USER $USERNAME

# install requirements
WORKDIR /src
RUN pip install --upgrade pip
COPY --chown=$USERNAME:$USERNAME requirements.txt requirements.txt
ENV PATH "${PATH}:/home/${USERNAME}/.local/bin"
RUN pip install --user -r requirements.txt
# RUN pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
# RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
