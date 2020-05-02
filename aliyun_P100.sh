#!/bin/sh

#Please input version to install
IS_INSTALL_PERSEUS="FALSE"
DRIVER_VERSION="410.104"
CUDA_VERSION="10.0.130"
CUDNN_VERSION="7.6.4"
IS_INSTALL_RAPIDS="FALSE"

INSTALL_DIR="/root/auto_install"
log=${INSTALL_DIR}/nvidia_install.log

#using .deb to install driver and cuda on ubuntu OS
#using .run to install driver and cuda on ubuntu OS
auto_install_script="auto_install.sh"

script_download_url=$(curl http://100.100.100.200/latest/meta-data/source-address | head -1)"/opsx/ecs/linux/binary/script/${auto_install_script}"
echo $script_download_url

mkdir $INSTALL_DIR && cd $INSTALL_DIR
wget -t 10 --timeout=10 $script_download_url && sh ${INSTALL_DIR}/${auto_install_script} $DRIVER_VERSION $CUDA_VERSION $CUDNN_VERSION $IS_INSTALL_PERSEUS $IS_INSTALL_RAPIDS
