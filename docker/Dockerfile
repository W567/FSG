FROM ros:noetic-ros-core-focal

ENV DEBIAN_FRONTEND=noninteractive

# install essential packages
RUN apt update && apt install -q -y --no-install-recommends \
    git \
    wget \
    pybind11-dev \
    build-essential \
    libcgal-dev \
    libopencv-dev \
    ros-noetic-urdf \
    ros-noetic-xacro \
    ros-noetic-eigen-conversions \
    ros-noetic-tf-conversions \
    ros-noetic-pcl-ros \
    && rm -rf /var/lib/apt/lists/*

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-rosinstall \
    python3-catkin-tools \
    python3-wstool \
    python-is-python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0-linux-x86_64.sh &&\
    chmod +x cmake-3.29.0-linux-x86_64.sh &&\
    mkdir -p /opt/cmake &&\
    ./cmake-3.29.0-linux-x86_64.sh --skip-license --prefix=/opt/cmake && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    rm cmake-3.29.0-linux-x86_64.sh
    
ENV PATH="/opt/cmake/bin:${PATH}"

RUN cd ~/ && git clone https://github.com/isl-org/Open3D.git -b v0.18.0 &&\
    cd ~/Open3D && util/install_deps_ubuntu.sh assume-yes &&\
    mkdir build && cd build && cmake .. && make -j && make install && make python-package &&\
    cd lib/python_package && pip install .

RUN pip install \
        h5py \
        lxml \
        sympy \
        mujoco \
        trimesh \
        scikit-robot \
        matplotlib
RUN pip uninstall -y pyrender && pip install git+https://github.com/mmatl/pyrender.git --no-cache-dir

RUN mkdir -p ~/catkin_ws/src
RUN cd ~/catkin_ws/src && git clone https://github.com/W567/FSG.git
RUN cd ~/catkin_ws/src && source /opt/ros/noetic/setup.bash &&\
    wstool init . && wstool merge -t . FSG/.fsg.rosinstall && wstool update -t .
RUN cd ~/catkin_ws && source /opt/ros/noetic/setup.bash &&\
    catkin init && catkin build fsg
RUN cd ~/catkin_ws/src/curvature_computation &&\
    mkdir build && cd build && cmake .. && make -j && make install
RUN cd ~/catkin_ws/src/pc_annotation &&\
    mkdir build && cd build && cmake .. && make -j && make install
RUN cd ~/catkin_ws/src/urdfpy && pip install -e .

CMD ["bash"]
