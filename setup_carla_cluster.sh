#!/bin/bash
# Install CARLA 0.9.15
# TODO: modify CARLA_ROOT if needed

CARLA_ROOT="${CARLA_ROOT:-/mnt/localssd/workspace/carla0915}"

mkdir -p "${CARLA_ROOT}"
cd "${CARLA_ROOT}"
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz
rm CARLA_0.9.15.tar.gz
cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh
